'''Single-node, multi-GPU CIFAR-10 training.'''
from pprint import pprint

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from main import build_model, fit, make_loaders, parse_args


def setup(local_rank, world_size):
    # Init process group
    print("Initialize Process Group...")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456',
                            rank=local_rank, world_size=world_size)


def worker(args):
    device = torch.device("cuda", dist.get_rank())
    net = build_model(args).to(device)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device.index], output_device=device.index)
    trainloader, testloader, sampler = make_loaders(args, device, distributed=True)
    fit(args, net, trainloader, testloader, sampler)


def ddp_worker(local_rank, world_size, args):
    setup(local_rank, world_size)
    try:
        worker(args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise SystemExit("main_dist.py requires CUDA. Use main.py for CPU training.")
    if args.batch_size < n_gpu:
        raise SystemExit("Batch size must be at least the number of GPUs.")
    pprint(vars(args))
    mp.spawn(ddp_worker, nprocs=n_gpu, args=(n_gpu, args))
