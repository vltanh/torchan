from PIL import Image
import numpy as np
import torch
from torchvision import transforms as tf

from os import listdir
from os.path import join


class PennFudanPed():
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(listdir(join(root, 'PNGImages'))))

    def __getitem__(self, idx):
        img_path = join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = img_path.replace('PNGImages', 'PedMasks') \
                            .replace('.png', '_mask.png')

        img = Image.open(img_path).convert('RGB')
        img_tf = tf.Compose([
            tf.ToTensor(),
        ])
        img = img_tf(img)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        nobjs = len(obj_ids)

        masks = mask == obj_ids[:, None, None]
        boxes = []
        for i in range(nobjs):
            x, y = np.where(masks[i])
            xmin, xmax = np.min(y), np.max(y)
            ymin, ymax = np.min(x), np.max(x)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((nobjs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((nobjs,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
        }
        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ds = PennFudanPed('data/PennFudanPed')
    for img, target in ds:
        bboxes = target['boxes']
        masks = target['masks']

        plt.imshow(img)
        ax = plt.gca()

        mask = torch.zeros(masks.shape[-2:])
        for c in range(masks.size(0)):
            mask[masks[c].bool()] = c + 1
        ax.imshow(mask, alpha=0.5)

        for bbox in bboxes:
            ax.add_patch(Rectangle((bbox[1], bbox[0]),
                                   bbox[3] - bbox[1],
                                   bbox[2] - bbox[0],
                                   fill=False,
                                   linewidth=1.5,
                                   edgecolor='r'
                                   ))

        plt.show()
        plt.close()
