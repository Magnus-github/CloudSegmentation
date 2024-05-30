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
                 transform: str = None,
                 image_size: int = 224,
                 overlap: int = 30,
                 split: str = "train",
                 random_state: int = 42) -> None:
        super().__init__()
        self._scenes_folder = os.path.join(data_folder, scenes_folder)
        self._masks_folder = os.path.join(data_folder, masks_folder)

        self._transform = str_to_class(transform)(image_size, overlap) if transform else BaseTransform()

        all_scene_files = sorted(os.listdir(self._scenes_folder))
        all_mask_files = sorted(os.listdir(self._masks_folder))

        train_idx, val_test_idx = train_test_split(range(len(all_scene_files)), test_size=0.15, random_state=random_state)
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.33, random_state=random_state)

        if split == "train": # 85% of the data
            self._scenes_files = [all_scene_files[i] for i in train_idx]
            self._masks_files = [all_mask_files[i] for i in train_idx]
        elif split == "val": # 10% of the data
            self._scenes_files = [all_scene_files[i] for i in val_idx]
            self._masks_files = [all_mask_files[i] for i in val_idx]
        elif split == "test": # 5% of the data
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
        # normalize to [0, 1]
        scene = (scene - scene.min()) / (scene.max() - scene.min())
        mask = np.load(os.path.join(self._masks_folder, mask_file))
        # merge 'CLEAR' and 'CLOUD_SHADOW' classes and keep only 'CLOUD' class
        clear_mask = mask[:,:,0] + mask[:,:,2]
        mask[:,:,0] = clear_mask
        mask = mask[:,:,1][:,:,None]

        scene = torch.tensor(scene.transpose(2,0,1), dtype=torch.float32)
        mask = torch.tensor(mask.transpose(2,0,1), dtype=torch.long)

        tiled_tf_scene, tiled_tf_mask = self._transform(image=scene, mask=mask)

        return tiled_tf_scene, tiled_tf_mask


def collate_fn(batch):
    scenes, masks = zip(*batch)
    scenes = torch.cat(scenes, dim=0)
    masks = torch.cat(masks, dim=0).squeeze(1)
    return scenes, masks


def get_dataloaders(cfg: omegaconf.DictConfig, fold: int = 0) -> dict[str, DataLoader]:
    train_dataset = Clouds(**cfg.dataset.params, **cfg.dataset.train_params)
    val_dataset = Clouds(**cfg.dataset.params, **cfg.dataset.val_params)
    test_dataset = Clouds(**cfg.dataset.params, **cfg.dataset.test_params)
    for f_train in train_dataset._scenes_files:
        for f_val in val_dataset._scenes_files:
            assert f_train != f_val, f"File {f_train} is in both train and val datasets"
        for f_test in test_dataset._scenes_files:
            assert f_train != f_test, f"File {f_train} is in both train and test datasets"
    for f_val in val_dataset._scenes_files:
        for f_test in test_dataset._scenes_files:
            assert f_val != f_test, f"File {f_val} is in both val and test datasets"
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.hparams.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.hparams.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloaders = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}
    return dataloaders


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