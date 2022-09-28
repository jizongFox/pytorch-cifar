'''Single node, multi-GPUs training.'''
from pprint import pprint
from turtle import shape
from torch import Tensor
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from loguru import logger
import sys
from models import *
from utils import progress_bar, IteratorTimer, item2str
from cifar10 import CIFAR10
import typing as t
import helper
import numpy as np

loguru_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <green>process-{process.id}</green> |<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'

logger.remove()
logger.add(sys.stderr,
           format=loguru_format,
           colorize=True,
           diagnose=False,
           backtrace=False)


def get_rank() -> int:
    try:
        return dist.get_rank()
    except RuntimeError:
        return 0


def get_world_size() -> int:
    try:
        return dist.get_world_size()
    except RuntimeError:
        return 1


class DistributedEMA:
    def __init__(
        self,
        *,
        num_samples: int,
        num_classes: int,
        true_labels: t.Sequence[Tensor],
        device: torch.device = torch.device("cuda"),
        dtype:torch.dtype= torch.float,
        alpha: float = 0.9,
        load_label=True,
        distributed=True,
    ):
        """
        EMA update across different process.

        Args:
            num_samples (_type_): _description_
            num_classes (_type_): _description_
            true_labels (_type_): _description_
            device (torch.device): _description_
            dtype (_type_, optional): _description_. Defaults to torch.dtype.
            alpha (float, optional): _description_. Defaults to 0.9.
        """
        self.__history = torch.zeros(num_samples,
                                    num_classes,
                                    device=device,
                                    dtype=dtype)
        self.alpha = alpha
        self._num_classes = num_classes
        self.device = device
        self.dtype = dtype
        assert 0 <= alpha <= 1, f"alpha must be between 0 and 1, given {alpha}."

        self.distributed = distributed

        if load_label:
            true_labels = torch.from_numpy(np.asarray(true_labels)).to(self.device)
            true_labels= helper.class2one_hot(true_labels, C=num_classes, class_dim=1)
            self.__history = true_labels.to(self.dtype)

        else:
            raise NotImplementedError()
        
        if self.distributed:
            self.num_process = dist.get_world_size()
        else:
            self.num_process = 1

    def add(self, prediction: Tensor, index: Tensor) -> None:
        b, c = prediction.shape
        assert c == self._num_classes, (c, self._num_classes)
        if self.distributed:
            self.__history[index.long()] = self.__history[index.long()] * (
                self.alpha * self.num_process -self.num_process + 1) + prediction.detach().to(self.device).type(
                    self.dtype) * self.num_process * (1 - self.alpha)
        else:
            self.__history[index.long()] = self.__history[index.long()] * (
                self.alpha) + prediction.detach().to(self.device).type(
                    self.dtype) * (1 - self.alpha)

    def update(self):
        if self.distributed:
            dist.all_reduce(self.__history, op=dist.ReduceOp.AVG)
        else:
            pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.update()
    @property
    def ema(self)-> Tensor:
        return self.__history

def setup(local_rank, world_size):
    # Init process group
    logger.info("Initialize Process Group...")
    dist.init_process_group(backend='nccl',
                            init_method='tcp://localhost:23456',
                            rank=local_rank,
                            world_size=world_size)
    torch.cuda.set_device(local_rank)
    # if local_rank>0:
    #     logger.remove()


def init(args):
    # Init Model
    logger.info("Initialize Model...")
    net = VGG("VGG11").cuda()
    if args.distributed:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[get_rank()], output_device=get_rank())
    else:
        net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(),
                                1e-3,
                                momentum=0.9,
                                weight_decay=1e-4)

    # Data
    logger.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root='./data',
                       train=True,
                       download=True,
                       transform=transform_train)
    trainsampler = None
    if args.distributed:
        trainsampler = torch.utils.data.distributed.DistributedSampler(
            trainset, shuffle=False)
    trainloader = DataLoader(trainset,
                             batch_size=args.batch_size // get_world_size(),
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             sampler=trainsampler,
                             persistent_workers=True)

    testset = CIFAR10(root='./data',
                      train=False,
                      download=True,
                      transform=transform_test)
    testloader = DataLoader(testset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=False,
                            persistent_workers=True)
    return net, optimizer, trainloader, testloader, criterion, trainsampler


def train(
    epoch,
    net,
    trainloader,
    optimizer,
    *,
    ema_prediction:DistributedEMA,
    args,
):
    logger.info('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    timer = IteratorTimer()
    for batch_idx, (inputs, targets, index) in enumerate(trainloader):
        with timer:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            if get_rank()==0:
                logger.info(f"\n{ema_prediction.ema[:10]}")
            
            with ema_prediction:
                ema_prediction.add(outputs.softmax(1), index)
                
            logger.info(f"softmax:\n, {outputs.softmax(1)[:10]}")
                
            if get_rank()==0:
                logger.info(f"\n{ema_prediction.ema[:10]}")
            breakpoint()
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


        if get_rank() == 0:
            progress_bar(
                batch_idx, len(trainloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (train_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))
    logger.info(f"Epoch {epoch}: {item2str(timer.summary())}")


@torch.no_grad()
def _test(
    epoch,
    net,
    testloader,
):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, index) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if get_rank() == 0:

            progress_bar(
                batch_idx, len(testloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))


def worker(args):
    net, optimizer, trainloader, testloader, criterion, trainsampler = init(
        args)

    #
    logger.info("intializing prediction matrix")
    ema_matrix = DistributedEMA(
        num_samples=len(trainloader.dataset),
        num_classes=trainloader.dataset.num_classes,
        true_labels=trainloader.dataset.targets,
        alpha=0.5, load_label=True, distributed=args.distributed,
        
    )
    logger.info("finished prediction matrix")

    for epoch in range(200):
        if args.distributed:
            trainsampler.set_epoch(epoch)
        train(epoch,
              net,
              trainloader,
              optimizer,
                ema_prediction=ema_matrix,
              args=args)
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
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    pprint(vars(args))
    if n_gpu > 1:
        args.distributed = True
        mp.spawn(ddp_worker, nprocs=n_gpu, args=(n_gpu, args))
    else:
        args.distributed = False

        worker(args)
