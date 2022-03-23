import typing as t
from abc import ABCMeta, abstractmethod

import torch
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from utils import AverageValueMeter, Metric, EMAMeter, item2str

if t.TYPE_CHECKING:
    from torch.optim import Optimizer

_default_lr_choices = [0.0001, 0.001, 0.01, 0.1]


class Strategy(metaclass=ABCMeta):
    def __init__(self, *, optimizer: "Optimizer", **kwargs) -> None:
        super().__init__()
        self.optimizer = optimizer
        self._ema_loss_meter, self._ema_acc_meter = self.config_meters(**kwargs)
        self._n = 0

    def config_meters(self, **kwargs) -> t.Tuple[Metric, Metric]:
        ema_loss_meter = AverageValueMeter()
        ema_acc_meter = AverageValueMeter()
        return ema_loss_meter, ema_acc_meter

    @abstractmethod
    def step(self, batch_accuracy: Tensor, batch_loss: Tensor) -> None:
        # set optimizer learning rate.
        pass

    def reset(self):
        self._ema_loss_meter.reset()
        self._ema_acc_meter.reset()

    def set_learning_rate(self, lr: float):
        logger.trace(f"{self.__class__.__name__} set learning rate to {lr:.3e}")
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class DoNothingStrategy(Strategy):

    def step(self, batch_accuracy: Tensor, batch_loss: Tensor) -> None:
        pass


class StochasticSampling(Strategy):
    _ema_loss_meter: t.List[EMAMeter]
    _ema_acc_meter: t.List[EMAMeter]

    def __init__(self, optimizer: "Optimizer", *, alpha: float = 0.9, T: float = 1.0,
                 learning_rate_choices: t.List[float] = None, update_every_step: int = 1) -> None:
        self._T = T
        self.learning_rate_choice = learning_rate_choices or _default_lr_choices
        super(StochasticSampling, self).__init__(optimizer=optimizer, alpha=alpha)
        self._update_every_step = update_every_step

    @property
    def choice_length(self) -> int:
        return len(self.learning_rate_choice)

    def config_meters(self, *, alpha: float) -> t.Tuple[t.List[Metric], t.List[Metric]]:
        ema_loss_meter = [EMAMeter(alpha=alpha) for _ in range(self.choice_length)]
        ema_acc_meter = [EMAMeter(alpha=alpha) for _ in range(self.choice_length)]
        return ema_loss_meter, ema_acc_meter

    def step(self, batch_accuracy: Tensor, batch_loss: Tensor) -> None:
        cur_learning_rate = self.get_learning_rate()
        metric_index = self.learning_rate_choice.index(cur_learning_rate)
        ema_loss_meter = self._ema_loss_meter[metric_index]
        ema_acc_meter = self._ema_acc_meter[metric_index]

        ema_loss_meter.add(batch_loss)
        ema_acc_meter.add(batch_accuracy)
        if (self._n % self._update_every_step) == 0:
            next_learning_rate = self.sample()
            self.set_learning_rate(next_learning_rate)
        self._n += 1

    def sample(self) -> float:
        probabilities = self.probs
        logger.trace(
            f"accuracy   : {item2str({f'{k:.1e}': v for k, v in zip(self.learning_rate_choice, self.accuracy_sequence)})}")
        logger.trace(
            f"probability: {item2str({f'{k:.1e}': v for k, v in zip(self.learning_rate_choice, probabilities.tolist()[0])})}")
        sampled_lr = self.learning_rate_choice[int(torch.multinomial(probabilities, num_samples=1))]

        return sampled_lr

    @property
    def accuracy_sequence(self):
        try:
            accuracy_list = [self._ema_acc_meter[i].summary() for i in range(self.choice_length)]
        except RuntimeError:  # the beginning would have runtime error since many learning rate are with empty results.
            accuracy_list = [1.0 for _ in range(self.choice_length)]
        return accuracy_list

    @property
    def probs(self):
        probabilities = F.softmax(
            torch.FloatTensor(self.accuracy_sequence).resize(1, self.choice_length) / (self._T + 1e-10), 1)
        return probabilities
