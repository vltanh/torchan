import torch
import pandas as pd

__all__ = ['MNISTDataset']


class MNISTDataset():
    def __init__(self, csv_path, is_rgb=False, is_train=True):
        df = pd.read_csv(csv_path, header=None)
        self.data = df.loc[:, 1:].values.reshape(-1, 1, 28, 28)
        self.labels = df.loc[:, 0].values
        self.is_train = is_train

        if is_rgb:
            self.data = self.data.repeat(3, 1)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.data[i]) / 255.
        lbl = self.labels[i]
        return img, lbl

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = MNISTDataset('data/MNIST/mnist_test.csv')
    print(len(ds))
    for i, (im, lbl) in enumerate(ds):
        print(im.shape, lbl)
        break

    ds = MNISTDataset('data/MNIST/mnist_test.csv', is_rgb=True)
    print(len(ds))
    for i, (im, lbl) in enumerate(ds):
        print(im.shape, lbl)
        break
