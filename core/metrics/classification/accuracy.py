import torch


__all__ = ['Accuracy']


class Accuracy():
    def __init__(self, *args, **kwargs):
        self.reset()

    def update(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        self.correct += correct
        self.sample_size += sample_size

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f'+ Accuracy: {self.value()}')
