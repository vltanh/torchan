import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from models import SpatialTransformerNet
from datasets import MNISTDataset
from utils.getter import get_instance, get_data

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

_, dataloader = get_data(config['config']['dataset'],
                         config['config']['seed'])

with torch.no_grad():
    for data, _ in dataloader:
        data = data.to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = torchvision.utils.make_grid(input_tensor)
        out_grid = torchvision.utils.make_grid(transformed_input_tensor)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(in_grid.permute(1, 2, 0))
        axes[0].set_title('Dataset Images')
        axes[1].imshow(out_grid.permute(1, 2, 0))
        axes[1].set_title('Transformed Images')

        plt.show()
        plt.close()
