import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class CloudSegmentationModel(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(CloudSegmentationModel, self).__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        # modify number of input channels
        self.model.backbone['0'][0] = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # modify the last layer to output the number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, 1)
        self.model.aux_classifier[4] = nn.Conv2d(10, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
