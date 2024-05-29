import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as DLV3_weights

from torchvision.models import resnet34, ResNet34_Weights



class CloudSegmentationModel(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
        super(CloudSegmentationModel, self).__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights="DEFAULT", weigts_backbone=DLV3_weights.DEFAULT)
        # modify number of input channels
        self.model.backbone['0'][0] = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # modify the last layer to output the number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, 1)
        self.model.aux_classifier[4] = nn.Conv2d(10, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Unet see: https://segmentation-modelspytorch.readthedocs.io/en/latest/#installation
# class CloudUnet(nn.Module):
#     def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
#         super(CloudUnet, self).__init__()
#         self.model = smp.Unet(
#             encoder_name="mobilenet_v2",
#             encoder_weights="imagenet",
#             in_channels=in_channels,
#             classes=num_classes,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)


class UNET(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        
        # Modify first layer of ResNet34 to accept custom number of channels
        base_model = resnet34(weights=ResNet34_Weights.DEFAULT) # Change this line
        base_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        
        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256*2, 128)
        self.upconv2 = self.expand_block(128*2, 64)
        self.upconv1 = self.expand_block(64*2, 64)
        self.upconv0 = self.expand_block(64*2, num_classes)
        

    def expand_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        expand = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        return expand


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Contracting Path
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Expansive Path
        upconv4 = self.upconv4(layer4)
        upconv3 = self.upconv3(torch.cat([upconv4, layer3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, layer2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, layer1], 1))
        upconv0 = self.upconv0(torch.cat([upconv1, layer0], 1))

        return upconv0

# https://www.kaggle.com/code/cordmaur/38-cloud-simple-unet
class UNET_2(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, num_classes, 3, 1)

    def contract_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1
