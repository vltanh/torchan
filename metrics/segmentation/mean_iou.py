import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['MeanIoU']


class MeanIoU():
    def __init__(self, nclasses, ignore_index=None, eps=1e-9):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def update(self, output, target):
        nclasses = output.size(1)
        prediction = torch.argmax(output, dim=1)
        prediction = F.one_hot(prediction, nclasses).bool()
        target = F.one_hot(target, nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        union = (prediction | target).sum((-3, -2))

        intersection = intersection.cpu()
        union = union.cpu()

        self.intersection += intersection.sum(0)
        self.union += union.sum(0)
        self.sample_size += intersection.size(0)

    def value(self):
        ious = (self.intersection + self.eps) / (self.union + self.eps)
        miou = ious.sum()
        nclasses = ious.size(0)
        if self.ignore_index is not None:
            miou -= ious[self.ignore_index]
            nclasses -= 1
        return miou / nclasses

    def reset(self):
        self.intersection = torch.zeros(self.nclasses).float()
        self.union = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = (self.intersection + self.eps) / (self.union + self.eps)

        print(f'mIoU: {self.value():.6f}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.6f}')
