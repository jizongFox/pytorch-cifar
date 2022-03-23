'''Single node, multi-GPUs training.'''
import argparse
from pprint import pprint

import torchvision
import torchvision.transforms as transforms
from loguru import logger
from torch.cuda.amp import GradScaler, autocast as AutoCast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from marco_idea import StochasticSampling, Strategy, DoNothingStrategy
from models import *
from utils import progress_bar, IteratorTimer, AverageValueMeter, LRCounter, adding_writable_sink, \
    config_logger, item2str


def init(args: argparse.Namespace):
    # Init Model
    print("Initialize Model...")
    net = ResNet18().cuda()
    net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)

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

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=False, persistent_workers=True,
        drop_last=True)
    scaler = GradScaler(enabled=args.enable_scale)
    autocast = AutoCast(enabled=args.enable_scale)
    if args.strategy_name == "constant":
        strategy = DoNothingStrategy(optimizer=optimizer)
    elif args.strategy_name == "stochastic":
        strategy = StochasticSampling(
            optimizer=optimizer, alpha=0.95, T=args.T, learning_rate_choices=sorted(set([*args.possible_lrs, args.lr])),
            update_every_step=10,
        )
    else:
        raise NotImplementedError(args.strategy_name)
    return net, optimizer, train_loader, test_loader, criterion, scaler, autocast, strategy


def train_epoch(epoch, net, train_loader, optimizer, *, scaler: GradScaler, autocast: AutoCast, strategy: Strategy):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = AverageValueMeter()
    accuracy_meter = AverageValueMeter()
    timer = IteratorTimer()
    lr_counter = LRCounter()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        with timer:
            inputs, targets = inputs.cuda(), targets.cuda()
            with autocast:
                outputs = net(inputs)
                loss = F.cross_entropy(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.add(loss.detach(), n=inputs.size(0))  # last batch loss
            predicted = outputs.argmax(1)
            batch_accuracy = predicted.eq(targets).float().mean().detach()  # last batch accuracy
            accuracy_meter.add(batch_accuracy, n=predicted.size(0))

            strategy.step(batch_accuracy=batch_accuracy, batch_loss=loss)
            lr_counter.add(strategy.get_learning_rate())

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%.3f%%)'
                     % (loss_meter.summary(), 100. * accuracy_meter.summary(), batch_accuracy * 100.))
    return {"accuracy": accuracy_meter.summary(), "loss": loss_meter.summary(), "lr_state": lr_counter.summary(),
            **timer.summary()}


@torch.no_grad()
def test_epoch(epoch, net, test_loader, *, scaler: GradScaler, autocast: AutoCast):
    net.eval()
    loss_meter = AverageValueMeter()
    accuracy_meter = AverageValueMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with autocast:
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

        loss_meter.add(loss.detach(), n=inputs.size(0))
        predicted = outputs.argmax(1)
        batch_accuracy = predicted.eq(targets).float().mean()
        accuracy_meter.add(batch_accuracy, n=predicted.size(0))

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%.3f%%)'
                     % (loss_meter.summary(), 100. * accuracy_meter.summary(), batch_accuracy * 100.))
    return {"accuracy": accuracy_meter.summary(), "loss": loss_meter.summary()}


def worker(args):
    net, optimizer, train_loader, test_loader, criterion, scaler, autocast, strategy = init(args)

    with SummaryWriter(args.save_dir) as writer:
        for epoch in range(200):
            train_metrics = train_epoch(epoch, net, train_loader, optimizer, scaler=scaler, autocast=autocast,
                                        strategy=strategy)
            logger.info(f"Training Epoch {epoch}: {item2str(train_metrics)}")

            lr_stats = train_metrics.pop("lr_state")
            for stats, value in train_metrics.items():
                writer.add_scalar(tag=f"tra/{stats}", scalar_value=value, global_step=epoch)
            writer.add_scalars(main_tag="tra_lr", tag_scalar_dict=lr_stats, global_step=epoch)

            test_metrics = test_epoch(epoch, net, test_loader, scaler=scaler, autocast=autocast)
            logger.info(f"Testing Epoch {epoch}: {item2str(test_metrics)}")

            for stats, value in test_metrics.items():
                writer.add_scalar(tag=f"test/{stats}", scalar_value=value, global_step=epoch)

            if isinstance(strategy, StochasticSampling):
                ema_acc = {str(strategy.learning_rate_choice[i]): strategy._ema_acc_meter[i].summary() for i in
                           range(strategy.choice_length)}
                writer.add_scalars(main_tag="tra_running_average", tag_scalar_dict=ema_acc, global_step=epoch)

                tra_probs = dict(zip([str(x) for x in strategy.learning_rate_choice], strategy.probs.tolist()[0]))
                writer.add_scalars(main_tag="tra_probs", tag_scalar_dict=tra_probs, global_step=epoch)


if __name__ == '__main__':
    n_gpu = torch.cuda.device_count()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--strategy-name', choices=['constant', "stochastic"], type=str, help="strategy name")
    parser.add_argument('--lr', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--possible-lrs', type=float, default=[0.001], nargs='+', help="possible learning rate")
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch Size")
    parser.add_argument("--num-workers", type=int, default=12, help="Number of workers")
    parser.add_argument("--T", type=float, default=0.1, help="temperature for softmax")
    parser.add_argument("--enable-scale", default=False, action="store_true", help="enable amp training")
    parser.add_argument("--save-dir", required=True, type=str, help="save directory for tensorboard and logs.")
    args = parser.parse_args()
    pprint(vars(args))

    config_logger()
    adding_writable_sink(save_dir=args.save_dir)

    worker(args)
