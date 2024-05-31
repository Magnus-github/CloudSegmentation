import torch
from torch.utils.data import DataLoader
import logging
import os
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from collections import OrderedDict

import omegaconf
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.utils import str_to_class
from scripts.dataset import get_dataloaders


class Trainer:
    def __init__(self, cfg: omegaconf.DictConfig, dataloaders: dict[DataLoader]) -> None:
        self._cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = str_to_class(cfg.model.name)(**cfg.model.in_params)
        self.model.to(self.device)
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.test_dataloader = dataloaders["test"]
        self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.in_params)
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.in_params)
        self.scheduler = str_to_class(cfg.hparams.scheduler.name)(self.optimizer, **cfg.hparams.scheduler.in_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level=logging.INFO)
        if cfg.wandb.enable:
            self.run_wandb = wandb.init(project=cfg.wandb.project_name, name=cfg.wandb.run_name, config=dict(cfg))
            self.run_wandb.watch(self.model)
        else:
            self.run_wandb = None

        self.logger.info(f"Device: {self.device}")

    def train(self) -> None:
        self.model.train()
        columns = ["Image", "Ground truth mask", "Predicted mask"]
        viz_table = wandb.Table(columns=columns)
        best_val_loss = np.inf
        for epoch in range(self._cfg.hparams.epochs):
            running_loss = 0.0
            running_acc = 0.0
            for i, (images, masks) in enumerate(tqdm(self.train_dataloader)):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(images)
                if type(outputs) == OrderedDict:
                    outputs = outputs["out"]

                prob = outputs.sigmoid()

                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                acc = self.accuracy_metric(prob.argmax(1), masks)
                running_acc += acc

                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                if self.run_wandb:
                    self.run_wandb.log({"Train loss [batch]": running_loss / (i + 1)})

            train_loss = running_loss / len(self.train_dataloader)
            train_acc = running_acc / len(self.train_dataloader)
            val_loss, val_acc, val_iou = self.validate(epoch,viz_table)

            self.logger.info(f"Epoch: {epoch}, Loss: {train_loss}, Accuracy: {train_acc}")
            self.logger.info(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
            if self.run_wandb:
                self.run_wandb.log(
                    {
                        "Train loss [epoch]": train_loss,
                        "Train Accuracy": train_acc,
                        "Validation loss": val_loss,
                        "Validation accuracy": val_acc,
                        "Validation IoU": val_iou,
                        "Learning rate": self.optimizer.param_groups[0]["lr"],
                        "Epoch": epoch,
                    }
                )            

            if val_loss < best_val_loss and val_loss < self._cfg.hparams.save_best_threshold:
                self.logger.info("Saving best model!")
                best_val_loss = val_loss
                self.save_model(best=True)

        results_dir = self._cfg.outputs.train_results
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{self._cfg.wandb.run_name+'_' if self._cfg.wandb.run_name else ''}results.txt"
        with open(os.path.join(results_dir, results_file), "a") as f:
            f.write(f"Model: {self._cfg.model.name}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}, Loss: {val_loss}, Accuracy: {val_acc}, IoU: {val_iou}\n")
        self.save_model()

        if self.run_wandb:
            self.run_wandb.finish()

    def validate(self,epoch: int = 0, viz_table: wandb.Table = None) -> tuple[float, float, float]:
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_acc = 0.0
            running_iou = 0.0
            for i, (images, masks) in enumerate(tqdm(self.val_dataloader)):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                if type(outputs) == OrderedDict:
                    outputs = outputs["out"]
                prob = outputs.sigmoid()
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                pred = prob.argmax(1)
                acc = self.accuracy_metric(pred, masks)
                running_acc += acc
                running_iou += self.compute_iou(pred, masks)

                if self.run_wandb and epoch % self._cfg.hparams.visualize_interval == 0 and i < 5:
                    self.logger.info("Visualizing!")
                    columns = ["Image", "Ground truth mask", "Predicted mask"]
                    viz_table = wandb.Table(columns=columns)
                    log_im = images[13,:3].permute(1, 2, 0).to("cpu").numpy()
                    log_mask = masks[13].to("cpu").numpy()
                    log_pred = pred[0].to("cpu").numpy()
                    viz_table.add_data(wandb.Image(log_im), wandb.Image(log_mask), wandb.Image(log_pred))
                    wandb.log({"test_predictions" : viz_table})
                    # figure = plt.figure(1)
                    # plt.subplot(131)
                    # plt.imshow(images[13,:3].permute(1, 2, 0).to("cpu").numpy())
                    # plt.title("Image")
                    # plt.axis("off")
                    # plt.subplot(132)
                    # plt.imshow(masks[13].to("cpu").numpy())
                    # plt.title("Ground truth mask")
                    # plt.axis("off")
                    # plt.subplot(133)
                    # plt.imshow(pred[0].to("cpu").numpy())
                    # plt.title("Predicted mask")
                    # plt.axis("off")

                    # self.run_wandb.log({"test_img_{i}".format(i=i): figure})
                    # plt.close()

            val_loss = running_loss / len(self.val_dataloader)
            val_acc = running_acc / len(self.val_dataloader)
            val_iou = running_iou / len(self.val_dataloader)

        self.model.train()

        return val_loss, val_acc, val_iou
    

    def test(self):
        self.model.eval()
        with torch.no_grad():
            running_acc = 0.0
            running_iou = 0.0
            for i, (images, masks) in enumerate(tqdm(self.test_dataloader)):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                if type(outputs) == OrderedDict:
                    outputs = outputs["out"]
                prob = outputs.sigmoid()
                pred = prob.argmax(1)
                acc = self.accuracy_metric(pred, masks)
                running_acc += acc
                running_iou += self.compute_iou(pred, masks)

                im = images[:5,:3].to("cpu")
                im -= im.min(1, keepdim=True)[0]
                im /= im.max(1, keepdim=True)[0]

                save_image(im, f"test_image.png")
                save_image(masks[:5].unsqueeze(1).to("cpu"), f"test_mask.png")
                save_image(pred[:5].unsqueeze(1).to("cpu"), f"test_pred.png")

            test_acc = running_acc / len(self.val_dataloader)
            test_iou = running_iou / len(self.val_dataloader)

        self.logger.info(f"Test Accuracy: {test_acc}, Test IoU: {test_iou}")

    def compute_iou(self, pred: torch.Tensor, true: torch.Tensor)  -> float:
        eps=1e-6
        pred = pred.int().cpu()
        true = true.int().cpu()
        intersection = (pred & true).float().sum((1, 2))
        union = (pred | true).float().sum((1, 2))
        
        iou = (intersection + eps) / (union + eps)
        
        return iou.mean().item()
    
    def accuracy_metric(self, pred: torch.Tensor, true: torch.Tensor) -> float:
        return (pred == true).float().mean().item()

    def save_model(self, best: bool = False, epoch: Optional[int] = None):
        if self.run_wandb:
            folder = self.run_wandb.name
        else:
            folder = "debug"
        model_name = self._cfg.model.name.split(":")[-1]
        name = model_name + "_final.pth"
        if best:
            name = model_name + "_best.pth"
        if epoch:
            name = model_name + f"_{epoch}.pth"

        save_path = os.path.join(self._cfg.model.save_path, folder)
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_path, name))


if __name__ == "__main__":
    import coloredlogs
    coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")
    cfg = omegaconf.OmegaConf.load("config/config.yaml")
    dataloaders = get_dataloaders(cfg)

    trainer = Trainer(cfg, dataloaders)
    trainer.train()
