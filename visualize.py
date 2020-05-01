import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.getter import get_instance, get_data
from utils.device import move_to

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight')
parser.add_argument('--gpus', default=None)
args = parser.parse_args()

dev_id = 'cuda:{}'.format(args.gpus) \
    if torch.cuda.is_available() and args.gpus is not None \
    else 'cpu'
device = torch.device(dev_id)

config = torch.load(args.weight)

model = get_instance(config['config']['model']).to(device)
model.load_state_dict(config['model_state_dict'])
model.eval()

dataloader, _ = get_data(config['config']['dataset'],
                         config['config']['seed'])

with torch.no_grad():
    for inputs, _ in dataloader:
        outputs = model(move_to(inputs, device))
        for img, target, output in zip(*inputs, outputs):
            bboxes = output['boxes'][:1]
            masks = output['masks'][:1, 0]
            masks = masks >= 0.5

            plt.imshow(img.permute(1, 2, 0))
            ax = plt.gca()

            mask = torch.zeros(masks.shape[-2:])
            for c in range(masks.size(0)):
                mask[masks[c].bool()] = c + 1
            ax.imshow(mask, alpha=0.5)

            for bbox in bboxes:
                ax.add_patch(Rectangle((bbox[0], bbox[1]),
                                       bbox[2] - bbox[0],
                                       bbox[3] - bbox[1],
                                       fill=False,
                                       linewidth=1.5,
                                       edgecolor='r'
                                       ))

            plt.show()
            plt.close()
