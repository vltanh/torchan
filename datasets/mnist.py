from torch.utils import data
from torchvision import datasets, transforms


class MNISTDataset(data.Dataset):
    def __init__(self, train):
        self.ds = datasets.MNIST(root='data', train=train, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)
