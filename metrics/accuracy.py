import torch


class Accuracy():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        return (pred == target)

    def update(self, value):
        self.correct += value.sum()
        self.total += value.size(0)

    def value(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0.0
        self.total = 0.0

    def summary(self):
        print(f'Accuracy: {self.value():.6f}')
