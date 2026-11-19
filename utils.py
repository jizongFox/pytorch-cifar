'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import sys
import time
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import Union, Dict, final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = 0, 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class IteratorTimer:

    def __init__(self) -> None:
        super().__init__()
        self._iter_meter = AverageValueMeter()
        self._data_meter = AverageValueMeter()
        self._cur_time = time.time()
        self._end_time = time.time()

    def iter_begin(self):
        self._cur_time = time.time()
        self._data_meter.add(self._cur_time - self._end_time)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def iter_end(self):
        self._end.record()
        torch.cuda.synchronize()
        self._iter_meter.add(self._start.elapsed_time(self._end) / 1000)
        self._end_time = time.time()

    def summary(self):
        return {"d_time": self._data_meter.summary(), "r_time": self._iter_meter.summary()}

    def __enter__(self):
        self.iter_begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.iter_end()


class Metric(metaclass=ABCMeta):
    _initialized = False

    def __init__(self, threaded=False, use_deque=False) -> None:
        super().__init__()
        self._initialized = True

    @abstractmethod
    def reset(self):
        pass

    @final
    def add(self, *args, **kwargs):
        assert self._initialized, f"{self.__class__.__name__} must be initialized by overriding __init__"
        return self._add(*args, **kwargs)

    @abstractmethod
    def _add(self, *args, **kwargs):
        pass

    @final
    def summary(self):
        return self._summary()

    @abstractmethod
    def _summary(self) -> Union[Dict[str, float], float]:
        pass


class AverageValueMeter(Metric):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def _add(self, value, n=1):
        self.sum += value * n
        self.n += n

    def reset(self):
        self.sum = 0
        self.n = 0

    def _summary(self) -> float:
        # this function returns a dict and tends to aggregate the historical results.
        if self.n == 0:
            return np.nan
        return float(self.sum / self.n)


def is_float(v):
    """if v is a scalar"""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def is_iterable(v):
    """if v is an iterable, except str"""
    if isinstance(v, str):
        return False
    return isinstance(v, (list, tuple, dict))


def _float2str(v):
    """convert a scalar to float, in order to display"""
    v = float(v)
    if abs(float(v)) < 0.01 or abs(float(v)) >= 999:
        return f"{v:.2e}"
    return f"{v:.3f}"


def _least_item2str(v):
    if is_float(v):
        return _float2str(v)
    return f"{v}"


def _generate_pair(k, v):
    """generate str for non iterable k v"""
    return f"{k}:{_least_item2str(v)}"


def _dict2str(dictionary: dict):
    strings = []
    for k, v in dictionary.items():
        if not is_iterable(v):
            strings.append(_generate_pair(k, v))
        else:
            strings.append(f"{k}:[" + item2str(v) + "]")
    return ", ".join(strings)


def _iter2str(item: Iterable):
    """A list or a tuple"""
    return ", ".join(
        [_least_item2str(x) if not is_iterable(x) else item2str(x) for x in item]
    )


def item2str(item):
    if isinstance(item, dict):
        return _dict2str(item)
    return _iter2str(item)
