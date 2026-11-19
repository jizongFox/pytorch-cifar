'''Single node, multi-GPUs training.'''
from pprint import pprint

import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import *
from utils import progress_bar, IteratorTimer, item2str


def setup(local_rank, world_size):
    # Init process group
    print("Initialize Process Group...")
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456',
                            rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def init(args):
    # Init Model
    print("Initialize Model...")
    net = EfficientNetB0().cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[dist.get_rank()], output_device=dist.get_rank())

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), 1e-3, momentum=0.9, weight_decay=1e-4)

    # Data
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(trainsampler is None),
        num_workers=args.num_workers,
        pin_memory=True, sampler=trainsampler,
        persistent_workers=True

    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False, persistent_workers=True)
    return net, optimizer, trainloader, testloader, criterion, trainsampler


def train(epoch, net, trainloader, optimizer, ):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    timer = IteratorTimer()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with timer:
            inputs, targets = inputs.cuda(), targets.cuda()
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


def _test(epoch, net, testloader, ):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def worker(args):
    net, optimizer, trainloader, testloader, criterion, trainsampler = init(args)
    for epoch in range(200):
        trainsampler.set_epoch(epoch)
        train(epoch, net, trainloader, optimizer)
        _test(epoch, net, testloader)


def ddp_worker(randk, world_size, args):
    setup(randk, world_size)
    worker(args)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    import argparse

    n_gpu = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    args = parser.parse_args()
    pprint(vars(args))
    mp.spawn(ddp_worker, nprocs=n_gpu, args=(n_gpu, args))
