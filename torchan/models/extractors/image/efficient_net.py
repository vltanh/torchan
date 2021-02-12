from .image_extractor import ImageExtractor
from efficientnet_pytorch import EfficientNet


class EfficientNetExtractor(ImageExtractor):
    def __init__(self, version, use_pretrained=False, is_frozen=False):
        super().__init__()
        assert version in [f'b{v}' for v in range(9)], \
            f'Invalid version [{version}].'
        if use_pretrained:
            self.extractor = \
                EfficientNet.from_pretrained(f'efficientnet-{version}')
        else:
            self.extractor = \
                EfficientNet.from_name(f'efficientnet-{version}')
        self.feature_dim = self.extractor._fc.in_features

        if is_frozen:
            self.freeze()

    def get_feature_map(self, x):
        return self.extractor.extract_features(x)
