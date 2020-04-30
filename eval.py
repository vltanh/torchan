import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from models import SpatialTransformerNet
from datasets import MNISTDataset


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


model = SpatialTransformerNet().cuda()
model.load_state_dict(torch.load(
    'runs/Test-2020_04_30-12_46_31/best_metric_Accuracy.pth')['model_state_dict'])

dataset = MNISTDataset(train=False)
dataloader = DataLoader(dataset, batch_size=64)

with torch.no_grad():
    for data, _ in dataloader:
        data = data.cuda()

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

        plt.show()
        plt.close()
