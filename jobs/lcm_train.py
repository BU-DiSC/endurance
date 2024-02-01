#!/usr/bin/env python
from typing import Any, Optional
import logging
import os

from torch.utils.data import DataLoader
import numpy as np
import toml
import torch
import torch.optim as TorchOpt

from endure.lcm.data.dataset import LCMDataSet
from endure.lcm.model.builder import LearnedCostModelBuilder
from endure.lsm.types import STR_POLICY_DICT, Policy
from endure.util.losses import LossBuilder
from endure.util.lr_scheduler import LRSchedulerBuilder
from endure.util.optimizer import OptimizerBuilder
from endure.util.trainer import Trainer


class LCMTrainJob:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.log = logging.getLogger(self.config["log"]["name"])
        self.log.info("Running Training Job")

        self.job_cfg = config["job"]["LCMTrain"]
        lsm_design = STR_POLICY_DICT.get(self.config["lsm"]["design"], None)
        if lsm_design is None:
            raise TypeError(f"Invalid LSM design: {self.config['lsm']['design']}")
        self.lsm_design = lsm_design

    def _build_loss_fn(self) -> torch.nn.Module:
        choice = self.job_cfg["loss_fn"]
        self.log.info(f"Loss function: {choice}")

        loss = LossBuilder(self.config).build(choice)
        if loss is None:
            self.log.warn("Invalid loss function. Defaulting to MSE")
            loss = LossBuilder(self.config).build("MSE")
        assert loss is not None

        if self.job_cfg["use_gpu_if_avail"] and torch.cuda.is_available():
            loss.to("cuda")

        return loss

    def _build_model(self) -> torch.nn.Module:
        lsm_choice = STR_POLICY_DICT.get(self.config["lsm"]["design"], Policy.KHybrid)
        size_ratio_min = self.config["lsm"]["size_ratio"]["min"]
        size_ratio_max = self.config["lsm"]["size_ratio"]["max"]
        model_builder = LearnedCostModelBuilder(
            size_ratio_range=(size_ratio_min, size_ratio_max),
            max_levels=self.config["lsm"]["max_levels"],
            **self.config["lcm"]["model"],
        )
        model = model_builder.build_model(lsm_choice)
        if self.job_cfg["use_gpu_if_avail"] and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_optimizer(self, model) -> TorchOpt.Optimizer:
        builder = OptimizerBuilder(self.config)
        choice = self.job_cfg["optimizer"]

        return builder.build_optimizer(choice, model)

    def _build_scheduler(
        self, optimizer: TorchOpt.Optimizer
    ) -> Optional[TorchOpt.lr_scheduler._LRScheduler]:
        builder = LRSchedulerBuilder(self.config)
        choice = self.job_cfg["lr_scheduler"]

        return builder.build_scheduler(optimizer, choice)

    def _build_train(self) -> DataLoader:
        train_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.job_cfg["train"]["dir"],
        )
        self.log.debug(f"Train data dir: {train_dir}")
        self.log.debug(f"Training features: {self.config['lcm']['input_features']}")
        train_data = LCMDataSet(
            folder=train_dir,
            lsm_design=self.lsm_design,
            min_size_ratio=self.config["lsm"]["size_ratio"]["min"],
            max_size_ratio=self.config["lsm"]["size_ratio"]["max"],
            max_levels=self.config["lsm"]["max_levels"],
            shuffle=self.job_cfg["train"]["shuffle"],
        )
        train = DataLoader(
            train_data,
            batch_size=self.job_cfg["train"]["batch_size"],
            drop_last=self.job_cfg["train"]["drop_last"],
            num_workers=self.job_cfg["train"]["num_workers"],
            pin_memory=True,
        )

        return train

    def _build_test(self) -> DataLoader:
        test_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.job_cfg["test"]["dir"],
        )
        self.log.info(f"Test data: {test_dir}")
        test_data = LCMDataSet(
            folder=test_dir,
            lsm_design=self.lsm_design,
            min_size_ratio=self.config["lsm"]["size_ratio"]["min"],
            max_size_ratio=self.config["lsm"]["size_ratio"]["max"],
            max_levels=self.config["lsm"]["max_levels"],
            test=True,
            shuffle=self.job_cfg["test"]["shuffle"],
        )
        test = DataLoader(
            test_data,
            batch_size=self.job_cfg["test"]["batch_size"],
            drop_last=self.job_cfg["test"]["drop_last"],
            num_workers=self.job_cfg["test"]["num_workers"],
            # collate_fn=self._test_collate_fn,
            pin_memory=True,
        )

        return test

    def _make_save_dir(self) -> Optional[str]:
        self.log.info(f"Saving tuner in {self.job_cfg['save_dir']}")
        save_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.job_cfg["save_dir"],
        )
        try:
            os.makedirs(save_dir, exist_ok=False)
        except FileExistsError:
            return None

        # dump configuration file
        with open(os.path.join(save_dir, "endure.toml"), "w") as fid:
            toml.dump(self.config, fid)

        return save_dir

    def run(self) -> Optional[Trainer]:
        model_base_dir = self._make_save_dir()
        if model_base_dir is None:
            self.log.info("Model directory already exists, exiting...")
            return None

        model = self._build_model()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        train_data = self._build_train()
        test_data = self._build_test()
        loss_fn = self._build_loss_fn()
        disable_tqdm = self.config["log"]["disable_tqdm"]

        trainer = Trainer(
            log=self.log,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            test_data=test_data,
            scheduler=scheduler,
            max_epochs=self.job_cfg["max_epochs"],
            use_gpu_if_avail=self.job_cfg["use_gpu_if_avail"],
            base_dir=model_base_dir,
            disable_tqdm=disable_tqdm,
            no_checkpoint=self.job_cfg["no_checkpoint"],
        )

        trainer.run()

        return trainer


if __name__ == "__main__":
    from endure.data.io import Reader

    config = Reader.read_config("endure.toml")

    logging.basicConfig(
        format=config["log"]["format"], datefmt=config["log"]["datefmt"]
    )

    log = logging.getLogger(config["log"]["name"])
    log.setLevel(config["log"]["level"])

    a = LCMTrainJob(config)
    a.run()
