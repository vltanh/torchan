import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNN_ResNet50_FPN(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.nclasses = nclasses

        self.in_features = \
            self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            FastRCNNPredictor(self.in_features, self.nclasses)

        self.in_features_mask = \
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.hidden_layer = 256
        self.model.roi_heads.mask_predictor = \
            MaskRCNNPredictor(self.in_features_mask,
                              self.hidden_layer,
                              self.nclasses)

    def forward(self, x):
        inputs, targets = x
        return self.model(inputs, targets)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    model = MaskRCNN_ResNet50_FPN(2)
    model.eval()
    inputs = [torch.rand(3, 300, 400),
              torch.rand(3, 500, 400)]
    predictions = model(inputs)

    for inp, pred in zip(inputs, predictions):
        bboxes = pred['boxes']
        masks = pred['masks'].squeeze(1)[:5]

        plt.imshow(inp.permute(1, 2, 0))
        ax = plt.gca()

        mask = torch.zeros(masks.shape[-2:])
        for c in range(masks.size(0)):
            mask[masks[c].bool()] = c + 1
        ax.imshow(mask, alpha=0.5)

        for bbox in bboxes:
            ax.add_patch(Rectangle((bbox[1], bbox[0]),
                                   bbox[3] - bbox[1],
                                   bbox[2] - bbox[0],
                                   fill=False,
                                   linewidth=1.5,
                                   edgecolor='r'
                                   ))

        plt.show()
        plt.close()
