import torch
import torch.utils.data as data
import csv


class MNISTDataset(data.Dataset):
    def __init__(self, path):
        super().__init__()
        assert open(path, 'r')

        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            data = [list(map(int, x)) for x in reader]

        data = torch.Tensor(data)
        self.imgs = data[:, 1:]
        self.lbls = data[:, 0]

    def __getitem__(self, i):
        img = self.imgs[i].float().reshape(1, 28, 28) / 255.0
        lbl = self.lbls[i].long()
        return img, lbl

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = MNISTDataset('data/MNIST/mnist_train.csv')
    for img, lbl in dataset:
        plt.imshow(img.squeeze(0))
        plt.title(lbl.item())
        plt.show()
