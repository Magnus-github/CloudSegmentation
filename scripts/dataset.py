import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import numpy as np

import os
import omegaconf
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.utils import str_to_class
from scripts.transforms import BaseTransform


class Clouds(Dataset):
    def __init__(self, data_folder: str = "",
                 scenes_folder: str = "subscenes",
                 masks_folder: str = "masks",
                 num_folds: int = 5,
                 transform: str = None,
                 image_size: int = 224,
                 overlap: int = 112,
                 mean: list[float] = [0.485, 0.456, 0.406],
                 std: list[float] = [0.229, 0.224, 0.225],
                 split: str = "train",
                 fold: int = 0,
                 random_state: int = 42) -> None:
        super().__init__()
        self._scenes_folder = os.path.join(data_folder, scenes_folder)
        self._masks_folder = os.path.join(data_folder, masks_folder)

        self._transform = str_to_class(transform)(image_size, overlap, mean, std) if transform else BaseTransform

        all_scene_files = sorted(os.listdir(self._scenes_folder))
        all_mask_files = sorted(os.listdir(self._masks_folder))

        # skf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        # splits = list(skf.split(all_scene_files, all_mask_files))
        # train_idx, val_test_idx = splits[fold]
        train_idx, val_test_idx = train_test_split(range(len(all_scene_files)), test_size=0.15, random_state=random_state)
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.3, random_state=random_state)

        if split == "train":
            self._scenes_files = [all_scene_files[i] for i in train_idx]
            self._masks_files = [all_mask_files[i] for i in train_idx]
        elif split == "val":
            self._scenes_files = [all_scene_files[i] for i in val_idx]
            self._masks_files = [all_mask_files[i] for i in val_idx]
        elif split == "test":
            self._scenes_files = [all_scene_files[i] for i in test_idx]
            self._masks_files = [all_mask_files[i] for i in test_idx]
        elif split == "all":
            self._scenes_files = all_scene_files
            self._masks_files = all_mask_files

    def __len__(self) -> int:
        return len(self._scenes_files)
        
    def __getitem__(self, idx: int):
        scene_file = self._scenes_files[idx]
        mask_file = self._masks_files[idx]

        assert scene_file == mask_file, f"Scene file {scene_file} and mask file {mask_file} do not match"

        scene = np.load(os.path.join(self._scenes_folder, scene_file))
        # get only desired bands (RGB and NIR; bands 4, 3, 2, 8)
        scene = scene[:,:,[3,2,1,7]]
        mask = np.load(os.path.join(self._masks_folder, mask_file))
        # merge 'CLEAR' and 'CLOUD_SHADOW' classes and keep only 'CLOUD' class
        clear_mask = mask[:,:,0] + mask[:,:,2]
        mask[:,:,0] = clear_mask
        mask = mask[:,:,1][:,:,None]

        scene = torch.tensor(scene.transpose(2,0,1), dtype=torch.float32)
        mask = torch.tensor(mask.transpose(2,0,1), dtype=torch.float32)

        tiled_tf_scene, tiled_tf_mask = self._transform(image=scene, mask=mask)

        return tiled_tf_scene, tiled_tf_mask


def collate_fn(batch):
    scenes, masks = zip(*batch)
    scenes = torch.cat(scenes, dim=0)
    masks = torch.cat(masks, dim=0).squeeze(1)
    return scenes, masks


def get_dataloaders(cfg: omegaconf.DictConfig, fold: int = 0) -> dict[str, DataLoader]:
    train_dataset = Clouds(**cfg.dataset.params, **cfg.dataset.train_params,fold=fold)
    val_dataset = Clouds(**cfg.dataset.params, **cfg.dataset.val_params,fold=fold)
    test_dataset = Clouds(**cfg.dataset.params, **cfg.dataset.test_params,fold=fold)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.hparams.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.hparams.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloaders = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}
    return dataloaders

from tqdm import tqdm
def calculate_mean_std(data_folder: str):
    dataset = Clouds(data_folder=data_folder, split="all")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(mean.shape, std.shape)
    print(mean, std)
    return mean, std


if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load("config/config.yaml")
    dataloaders = get_dataloaders(cfg)

    for i, (scene, mask) in enumerate(dataloaders["train"]):
        print(scene.shape, mask.shape)

        save_image(scene[:5,:3], f"scene.png")
        save_image(mask[:5,0].unsqueeze(1).float(), f"mask_clear.png")
        save_image(mask[:5,1].unsqueeze(1).float(), f"mask_cloud.png")

        if i == 10:
            break