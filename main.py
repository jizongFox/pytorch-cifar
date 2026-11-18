'''CIFAR-10 training on CPU or CUDA.'''
import argparse
from pprint import pprint

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import EfficientNetB0, VisionTransformer
from utils import progress_bar, IteratorTimer, item2str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--model", choices=("efficientnet", "vit", "looped-vit"),
                        default="efficientnet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of encoder block applications for either ViT")
    parser.add_argument("--lr", type=float, help="Override the model's default learning rate")
    parser.add_argument("--seed", type=int, help="Seed model initialization and data shuffling")
    parser.add_argument("--eval-loop-steps", type=int, nargs="+", default=[],
                        help="Evaluate a trained looped ViT at these step counts after training")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.depth) < 1 or args.num_workers < 0:
        parser.error("batch size, epochs, and depth must be positive, and workers nonnegative")
    if args.lr is not None and not 0 < args.lr < float("inf"):
        parser.error("learning rate must be finite and positive")
    if args.eval_loop_steps:
        if args.model != "looped-vit":
            parser.error("--eval-loop-steps requires --model looped-vit")
        if any(not 1 <= steps <= args.depth for steps in args.eval_loop_steps):
            parser.error("--eval-loop-steps must be between 1 and --depth")
    return args


def build_model(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.model == "efficientnet":
        net = EfficientNetB0()
    else:
        net = VisionTransformer(depth=args.depth, looped=args.model == "looped-vit")
    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"{args.model} with {parameters:,} trainable parameters")
    return net


def make_loaders(args, device, distributed=False):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    rank = dist.get_rank() if distributed else 0
    # Let rank zero finish downloading before other ranks read the dataset.
    if rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=(rank == 0), transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=(rank == 0), transform=transform_test)
    if distributed and rank == 0:
        dist.barrier()

    generator = None
    if args.seed is not None:
        # Keep data order and augmentation independent of model initialization.
        torch.manual_seed(args.seed + rank)
        generator = torch.Generator().manual_seed(args.seed + rank)

    sampler = None
    batch_size = args.batch_size
    if distributed:
        sampler = DistributedSampler(trainset, seed=args.seed if args.seed is not None else 0)
        batch_size //= dist.get_world_size()
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0), generator=generator,
    )
    test_workers = min(2, args.num_workers)
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=test_workers,
        persistent_workers=(test_workers > 0),
    )
    return trainloader, testloader, sampler


def train(epoch, net, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    device = next(net.parameters()).device
    timer = IteratorTimer(device)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with timer:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print(f"Epoch {epoch}: {item2str(timer.summary())}")


def _test(net, testloader, steps=None):
    net.eval()
    print("\nTest" if steps is None else f"\nTest at {steps} loop steps")
    test_loss = 0
    correct = 0
    total = 0
    device = next(net.parameters()).device
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) if steps is None else net(inputs, steps=steps)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def fit(args, net, trainloader, testloader, sampler=None):
    scheduler = None
    if args.model == "efficientnet":
        optimizer = torch.optim.SGD(net.parameters(), args.lr or 1e-3,
                                    momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr or 3e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train(epoch, net, trainloader, optimizer)
        _test(net, testloader)
        if scheduler is not None:
            scheduler.step()

    for steps in dict.fromkeys(args.eval_loop_steps):
        if steps != args.depth:
            _test(net, testloader, steps=steps)


def worker(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model(args).to(device)
    if device.type == "cuda":
        net = nn.DataParallel(net)
    trainloader, testloader, _ = make_loaders(args, device)
    fit(args, net, trainloader, testloader)


if __name__ == '__main__':
    args = parse_args()
    pprint(vars(args))
    worker(args)
