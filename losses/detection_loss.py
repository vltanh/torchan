import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        if isinstance(output, list):
            return torch.tensor([0.0])
        return torch.sum(torch.stack(list(output.values())))
