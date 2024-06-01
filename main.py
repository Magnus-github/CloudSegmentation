import argparse

from scripts.dataset import get_dataloaders
from scripts.trainer import Trainer

import omegaconf
import logging
import coloredlogs
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")


def main(args: argparse.Namespace) -> None:
    cfg = omegaconf.OmegaConf.load(args.config)
    dataloaders = get_dataloaders(cfg)

    trainer = Trainer(cfg, dataloaders)

    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        cfg.model.load_weights = True
        trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    args = parser.parse_args()
    main(args)
