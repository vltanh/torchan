import torch
import torch.utils.data as data
import torchvision.transforms as tvtf
import csv

from PIL import Image
import os

class CIFAR10Dataset(data.Dataset):
    def __init__(self, img_dir, label_path=None):
        super().__init__()

        self.img_dir = img_dir

        self.cid2name = [ 
            'airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        if label_path is not None:
            with open(label_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                data = [(int(_id), self.cid2name.index(name)) for _id, name in reader]
            self.ids, self.lbls = list(zip(*data))
            self.is_train = True
        else:
            self.ids = [int(os.path.splitext(x)[0]) for x in os.listdir(img_dir)]
            self.is_train = False

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, f'{self.ids[i]}.png'))
        img = tvtf.Compose([
            tvtf.ToTensor(),
        ])(img)
        
        if self.is_train:
            lbl = torch.Tensor([self.lbls[i]]).long().squeeze()
            return img, lbl
        else:
            return img, self.ids[i]

    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CIFAR10Dataset('data/CIFAR10/train', 
                             'data/CIFAR10/trainLabels.csv')
    for img, lbl in dataset:
        plt.imshow(img.permute(1, 2, 0))
        plt.title(lbl.item())
        plt.show()