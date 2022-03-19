import torch

from core.utils import getter

__all__ = ['MultiClsMetric']


def take_last(output, target):
    return output[:, -1], target


def take_all(output, target):
    # output: B, R+1, C
    # target: B
    B, R, C = output.size()

    target = target.unsqueeze(0).repeat(R, 1).T  # R+1, B
    target = target.reshape(-1)  # (R+1)*B

    output = output.reshape(-1, C)  # B*(R+1), C

    return output, target


def take_mean(output, target):
    # output: B, R+1, C
    # target: B
    output = torch.softmax(output, dim=2)
    output = output.mean(dim=1)
    return output, target


class MultiClsMetric:
    def __init__(self, metric_cfg, strategy='last'):
        self.metric = getter.get_instance(metric_cfg)
        self.strategy = {
            'last': take_last,
            'all': take_all,
            'mean': take_mean,
        }[strategy]

        self.reset = self.metric.reset
        self.value = self.metric.value
        self.summary = self.metric.summary

    def update(self, output, target):
        return self.metric.update(*self.strategy(output, target))
