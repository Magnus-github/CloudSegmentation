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
        if cfg.model.load_weights.enable:
            self.model.load_state_dict(torch.load(cfg.model.load_weights.path, map_location=self.device))
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.test_dataloader = dataloaders["test"]
        self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.in_params)
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.in_params)
        self.scheduler = str_to_class(cfg.hparams.scheduler.name)(self.optimizer, **cfg.hparams.scheduler.in_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level=logging.INFO)
        if cfg.wandb.enable:
            self.run_wandb = wandb.init(project=cfg.wandb.project_name, config=dict(cfg))
            self.run_wandb.watch(self.model)
        else:
            self.run_wandb = None

        self.logger.info(f"Device: {self.device}")

    def train(self) -> None:
        self.model.train()
        best_val_loss = np.inf
        for epoch in range(self._cfg.hparams.epochs):
            running_loss = 0.0
            running_label = 0
            running_correct = 0
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

                # labeled, correct = self.compute_accuracy(masks, outputs, self._cfg.model.in_params.num_classes)
                # running_label += labeled.sum()
                # running_correct += correct

                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                if self.run_wandb:
                    self.run_wandb.log({"Train loss [batch]": running_loss / (i + 1)})

            train_loss = running_loss / len(self.train_dataloader)
            # train_acc = ((1.0 * running_correct) / (np.spacing(1) + running_label)) * 100
            train_acc = running_acc / len(self.train_dataloader)
            val_loss, val_acc, val_iou = self.validate()

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

        self.save_model()

        if self.run_wandb:
            self.run_wandb.finish()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_label = 0
            running_correct = 0
            running_acc = 0.0
            running_iou = 0
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
                # labeled, correct = self.compute_accuracy(masks, outputs, self._cfg.model.in_params.num_classes)
                # running_label += labeled.sum()
                # running_correct += correct
                # pred = outputs.softmax(1)[:,1]
                # pred = (prob > 0.5).float()
                running_iou += self.compute_iou(pred, masks)

                # if self.run_wandb and epoch % self._cfg.hparams.visualize_interval == 0 and i < 5:
                #     self.logger.info("Visualizing!")
                #     figure = plt.figure(1)
                #     plt.subplot(131)
                #     plt.imshow(images[0].permute(1, 2, 0).to("cpu"))
                #     plt.title("Image")
                #     plt.subplot(132)
                #     plt.imshow(masks[0].to("cpu"))
                #     plt.title("Ground truth mask")
                #     plt.subplot(133)
                #     plt.imshow(outputs[0].argmax(0).to("cpu"))
                #     plt.title("Predicted mask")

                #     self.run_wandb.log({"test_img_{i}".format(i=i): figure})
                #     plt.close()

            val_loss = running_loss / len(self.val_dataloader)
            # val_acc = ((1.0 * running_correct) / (np.spacing(1) + running_label)) * 100
            val_acc = running_acc / len(self.val_dataloader)
            val_iou = running_iou / len(self.val_dataloader)

        self.model.train()

        return val_loss, val_acc, val_iou
    

    def test(self):
        self.model.eval()
        with torch.no_grad():
            running_label = 0
            running_correct = 0
            for i, (images, masks) in enumerate(tqdm(self.test_dataloader)):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                if type(outputs) == OrderedDict:
                    outputs = outputs["out"]
                labeled, correct = self.compute_accuracy(masks, outputs, self._cfg.model.in_params.num_classes)
                running_label += labeled.sum()
                running_correct += correct
                prob = outputs.sigmoid()
                # pred = (prob > 0.5).float()
                pred = prob.argmax(1)

                im = images[:5,:3].to("cpu")
                im -= im.min(1, keepdim=True)[0]
                im /= im.max(1, keepdim=True)[0]

                save_image(im, f"test_image.png")
                save_image(masks[:5].unsqueeze(1).to("cpu"), f"test_mask.png")
                save_image(pred[:5].unsqueeze(1).to("cpu"), f"test_pred.png")

            test_acc = ((1.0 * running_correct) / (np.spacing(1) + running_label)) * 100

        self.logger.info(f"Test Accuracy: {test_acc}")

    def compute_iou(self, pred: torch.Tensor, true: torch.Tensor)  -> float:
        eps=1e-6
        pred = pred.int().cpu()
        true = true.int().cpu()
        intersection = (pred & true).float().sum((1, 2))
        union = (pred | true).float().sum((1, 2))
        
        iou = (intersection + eps) / (union + eps)
        
        # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
        
        return iou.mean().item()
    
    def accuracy_metric(self, pred: torch.Tensor, true: torch.Tensor) -> float:
        return (pred == true).float().mean().item()

    def compute_accuracy(self, target: torch.Tensor, outputs: torch.Tensor, num_classes: int):
        labeled = target.ne(0)
        pred = outputs.sigmoid()
        # pred = (pred > 0.5).float()
        pred = pred.argmax(1)
        correct = (pred == target).float()
        correct = (correct * labeled).sum()
        labeled = labeled.sum()

        return labeled, correct

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
