import torch
import numpy as np
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors


class BaseTransform:
    def  __init__(self, imsize: int = 224, overlap: float = 112, mean: list[float] = [0.485, 0.456, 0.406], std: list[float] = [0.229, 0.224, 0.225]) -> None:
        self._imsize = imsize
        self._stride = imsize - overlap
        self.normalize = transforms.Normalize(mean, std)
    
    def _tile(self, image: torch.Tensor) -> torch.Tensor:
        tilew = image.unfold(1,self._imsize,self._stride)
        tilewh = tilew.unfold(2,self._imsize,self._stride)
        c,tw,th,w,h = tilewh.shape
        tiled_im = tilewh.permute(1,2,0,3,4).reshape(tw*th,c,w,h)
        return tiled_im
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.normalize(image)
        tiled_im = self._tile(image)
        tiled_mask = self._tile(mask)
        return tiled_im, tiled_mask


class TrainTransform(BaseTransform):
    def __init__(self, imsize: int = 224, overlap: float = 112, mean: list[float] = [0.485, 0.456, 0.406], std: list[float] = [0.229, 0.224, 0.225]) -> None:
        super(TrainTransform, self).__init__(imsize, overlap, mean, std)
        self._transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=60),
                # transforms.RandomResizedCrop(imsize, scale=(0.8, 1.0)),
            ]
        )

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image = self.normalize(image)
        tiled_im = self._tile(image)
        tiled_mask = self._tile(mask)
        tiled_mask = tv_tensors.Mask(tiled_mask)

        return self._transform(tiled_im, tiled_mask)


class ValTransform(BaseTransform):
    def __init__(self, imsize: int = 224, overlap: float = 112, mean: list[float] = [0.485, 0.456, 0.406], std: list[float] = [0.229, 0.224, 0.225]) -> None:
        super(ValTransform, self).__init__(imsize, overlap, mean, std)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image = self.normalize(image)
        tiled_im = self._tile(image)
        tiled_mask = self._tile(mask)
        return tiled_im, tiled_mask