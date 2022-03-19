import torch
import torch.nn.functional as F

__all__ = ['DiceScore']


class DiceScore():
    def __init__(self, nclasses, ignore_index=None, eps=1e-6):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def update(self, output, target):
        batch_size = output.size(0)
        ious = torch.zeros(self.nclasses, batch_size)

        prediction = torch.argmax(output, dim=1)

        if self.ignore_index is not None:
            target_mask = (target == self.ignore_index).bool()
            prediction[target_mask] = self.ignore_index

        prediction = F.one_hot(prediction, self.nclasses).bool()
        target = F.one_hot(target, self.nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        total_count = (prediction.float() + target.float()).sum((-3, -2))
        ious = 2 * (intersection.float() + self.eps) / (total_count + self.eps)

        ious = ious.cpu()
        self.mean_class += ious.sum(0)
        self.sample_size += ious.size(0)

    def value(self):
        return (self.mean_class / self.sample_size).mean()

    def reset(self):
        self.mean_class = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = self.mean_class / self.sample_size
        dice_score = class_iou.mean()

        print(f'+ Dice Score: {dice_score:.6f}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.6f}')
