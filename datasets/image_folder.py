from PIL import Image

import os


class ImageFolderDataset:
    def __init__(self, img_dir, transforms):
        self.dir = img_dir
        self.filenames = os.listdir(img_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.dir, filename)
        im = Image.open(img_path).convert('RGB')
        im = self.transforms(im)
        return im, filename

    def __len__(self):
        return len(self.fns)
