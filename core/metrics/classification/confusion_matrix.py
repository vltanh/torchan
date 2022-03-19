import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

__all__ = ['ConfusionMatrix']


class ConfusionMatrix():
    def __init__(self, nclasses, print=True, savefig_dir=None):
        self.nclasses = nclasses
        self.print = print
        self.savefig_dir = savefig_dir
        self.reset()

    def update(self, output, target):
        pred = torch.argmax(output, dim=1)
        self.cm += confusion_matrix(target.cpu().numpy(),
                                    pred.cpu().numpy(),
                                    labels=range(self.nclasses))

    def reset(self):
        self.cm = np.zeros(shape=(self.nclasses, self.nclasses))

    def value(self):
        return None

    def summary(self):
        print('+ Confusion matrix: ')
        if self.print:
            print(self.cm)
        if self.savefig_dir is not None:
            df_cm = pd.DataFrame(self.cm,
                                 index=range(self.nclasses),
                                 columns=range(self.nclasses))
            plt.figure(figsize=(10, 7))
            sns.heatmap(df_cm, annot=True, cmap='YlGnBu')
            plt.tight_layout()
            plt.savefig(self.savefig_dir + '/cm.png')
