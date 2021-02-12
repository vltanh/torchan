from torch import nn


class Extractor(nn.Module):
    def freeze(self):
        for p in self.extractor.parameters():
            p.requires_grad = False

    def get_embedding(self, x):
        raise NotImplementedError
