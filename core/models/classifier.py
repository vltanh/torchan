import torch
import torch.nn as nn

from core.utils import getter

__all__ = ['BaseClassifier', 'MultiClassifier']


class BaseClassifier(nn.Module):
    def __init__(self, nclasses, extractor_cfg, classifier_cfg=None):
        super().__init__()
        self.nclasses = nclasses
        self.extractor = self.init_extractor(extractor_cfg)
        self.feature_dim = self.extractor.feature_dim
        self.classifier = self.init_classifier(classifier_cfg)

    def init_extractor(self, config):
        return getter.get_instance(config)

    def init_classifier(self, _):
        return nn.Linear(self.feature_dim, self.nclasses)

    def forward(self, x):
        x = self.extractor(x)
        return self.classifier(x)

    def get_embedding(self, x):
        return self.extractor.get_embedding(x)

    def get_logit_from_emb(self, embeddings):
        return self.classifier(embeddings)


class MultiClassifier(BaseClassifier):
    def init_classifier(self, config):
        return nn.ModuleList([
            nn.Linear(self.feature_dim, self.nclasses)
            for _ in range(config['nclassifiers'])
        ])

    def forward(self, x):
        # x: B, R, *
        x = self.extractor(x)  # B, R+1, D
        x = torch.cat([
            cls(x[:, i]).unsqueeze(1)  # B, 1, C
            for i, cls in enumerate(self.classifier)
        ], dim=1)  # B, R+1, C
        return x

    def get_embedding(self, x):
        # x: B, R, *
        return self.extractor.get_embedding(x)  # B, D

    def get_logit_from_emb(self, x):
        # x: B, D
        return self.classifier[-1](x)  # B, C
