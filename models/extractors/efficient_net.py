from .extractor_network import ExtractorNetwork
from efficientnet_pytorch import EfficientNet


class EfficientNetExtractor(ExtractorNetwork):
    def __init__(self, version):
        super().__init__()
        assert version in [f'b{id}' for id in range(7)]
        self.extractor = EfficientNet.from_pretrained(
            f'efficientnet-{version}')
        self.feature_dim = self.extractor._fc.in_features

    def forward(self, x):
        x = self.extractor.extract_features(x)
        x = self.extractor._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return x
