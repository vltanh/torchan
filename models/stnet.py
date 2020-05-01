import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Localisation network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                                    dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, output_size=(3, 3)).view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


class SpatialTransformerNet(nn.Module):
    def __init__(self, in_channels, feature_dim, nclasses):
        super().__init__()
        self.stn = SpatialTransformerModule(in_channels)
        self.conv = nn.Conv2d(in_channels, feature_dim, 3)
        self.fc = nn.Linear(feature_dim, nclasses)

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv(x), 2))
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        return self.fc(x)
