from torch.nn import functional as F

from core.utils import getter
from core.models.extractors.extractor import Extractor

__all__ = ['ImageMaskExtractor']


class ImageExtractor(Extractor):
    def get_feature_map(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.get_embedding(x)

    def get_embedding(self, x):
        x = self.get_feature_map(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageMaskExtractor(ImageExtractor):
    def __init__(self, ext_cfg):
        super().__init__()
        self.ext = getter.get_instance(ext_cfg)
        self.feature_dim = self.ext.feature_dim

    def get_feature_map(self, x):
        # x: B, C+1, H, W
        im = x[:, :-1]  # B, C, H, W
        mask = x[:, -1]  # B, H, W

        # Extract features from image
        ft_map = self.ext.get_feature_map(im)  # B, D, H', W'

        # Resize mask to match new image size
        mask = mask.unsqueeze(1).float()  # B, 1, H, W
        mask = F.interpolate(mask, size=ft_map.shape[-2:],
                             mode='bilinear', align_corners=True)  # B, 1, H', W'

        # Masking by element-wise multiplication
        ft_map *= mask  # B, D, H', W'

        return ft_map

    def get_embedding(self, x):
        # x: B, C+1, H, W
        im = x[:, :-1]  # B, C, H, W
        mask = x[:, -1]  # B, H, W

        # Get feature map
        x = self.get_feature_map(x)  # B, D, H', W'
        nh, nw = x.shape[-2:]
        mask = F.interpolate(mask.unsqueeze(1).float(), size=x.shape[-2:],
                             mode='bilinear', align_corners=True)  # B, 1, H', W'

        # Take the verage vector
        x = F.adaptive_avg_pool2d(x, (1, 1))  # B, D, 1, 1
        x = x.view(x.size(0), -1)  # B, D

        # Normalize
        x *= nh * nw
        fg_area = mask.sum((-1, -2))  # B, 1
        x /= fg_area  # B, D

        return x
