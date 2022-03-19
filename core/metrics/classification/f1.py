import torch
from sklearn.metrics import f1_score

__all__ = ['F1']


class F1():
    def __init__(self, nclasses, ignore_classes=None, mode='weighted'):
        self.mode = mode
        self.labels = list(range(nclasses))
        if ignore_classes is not None:
            self.labels = list(
                filter(lambda x: x not in ignore_classes,
                       self.labels)
            )
        self.reset()

    def update(self, output, target):
        pred = torch.argmax(output, dim=1)
        self.pred += pred.cpu().tolist()
        self.target += target.cpu().tolist()

    def reset(self):
        self.pred = []
        self.target = []

    def value(self):
        return f1_score(self.target, self.pred,
                        labels=self.labels, average=self.mode)

    def summary(self):
        print(f'+ F1:')

        for mode in ['micro', 'macro', 'weighted']:
            f1 = f1_score(self.target, self.pred,
                          labels=self.labels, average=mode)
            print(f'{mode}: {f1}')

        print(f'class:')
        f1 = f1_score(self.target, self.pred,
                      labels=self.labels, average=None)
        for c, s in zip(self.labels, f1):
            print(f'\t{c}: {s}')
