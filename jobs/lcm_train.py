#!/usr/bin/env python
from typing import Any, Optional
import logging
import os
import toml
import torch

from torch.utils.data import DataLoader
import torch.optim as TorchOpt

from endure.lcm.data.iterable_dataset import LCMIterableDataSet
from endure.lcm.data.classic_dataset import LCMDataSet
from endure.lcm.model.builder import LearnedCostModelBuilder
from endure.util.losses import LossBuilder
from endure.util.lr_scheduler import LRSchedulerBuilder
from endure.util.optimizer import OptimizerBuilder
from endure.util.trainer import Trainer


class LCMTrainJob:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._setting = config["job"]["LCMTrain"]
        self.log = logging.getLogger(self._config["log"]["name"])
        self.log.info("Running Training Job")

    def _build_loss_fn(self) -> torch.nn.Module:
        choice = self._setting["loss_fn"]
        self.log.info(f"Loss function: {choice}")

        loss = LossBuilder.build(choice)
        if loss is None:
            self.log.warn("Invalid loss func. Defaulting to MSE")
            loss = LossBuilder.build("MSE")
        assert loss is not None

        if self._setting["use_gpu_if_avail"] and torch.cuda.is_available():
            loss.to("cuda")

        return loss

    def _build_model(self) -> torch.nn.Module:
        model = LearnedCostModelBuilder(self._config).build_model()
        if self._setting["use_gpu_if_avail"] and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_optimizer(self, model) -> TorchOpt.Optimizer:
        builder = OptimizerBuilder(self._config)
        choice = self._setting["optimizer"]

        return builder.build_optimizer(choice, model)

    def _build_scheduler(
        self, optimizer: TorchOpt.Optimizer
    ) -> TorchOpt.lr_scheduler._LRScheduler:
        builder = LRSchedulerBuilder(self._config)
        choice = self._setting["lr_scheduler"]

        return builder.build_scheduler(optimizer, choice)

    def _build_train(self) -> DataLoader:
        train_dir = os.path.join(
            self._config["io"]["data_dir"],
            self._setting["train"]["dir"],
        )
        self.log.info(f"Train data: {train_dir}")

        train_data = LCMIterableDataSet(
            config=self._config,
            folder=train_dir,
            shuffle=self._setting["train"]["shuffle"],
            format=self._setting["train"]["format"],
        )
        # train_data = LCMDataSet(
        #     config=self._config,
        #     folder=train_dir,
        #     format=self._setting["train"]["format"],
        # )
        train = DataLoader(
            train_data,
            batch_size=self._setting["train"]["batch_size"],
            drop_last=self._setting["train"]["drop_last"],
            num_workers=self._setting["train"]["num_workers"],
            pin_memory=True,
        )

        return train

    def _build_test(self) -> DataLoader:
        test_dir = os.path.join(
            self._config["io"]["data_dir"],
            self._setting["test"]["dir"],
        )
        self.log.info(f"Test data: {test_dir}")

        test_data = LCMIterableDataSet(
            config=self._config,
            folder=test_dir,
            shuffle=self._setting["test"]["shuffle"],
            format=self._setting["test"]["format"],
        )
        # test_data = LCMDataSet(
        #     config=self._config,
        #     folder=test_dir,
        #     format=self._setting["test"]["format"],
        # )
        test = DataLoader(
            test_data,
            batch_size=self._setting["test"]["batch_size"],
            drop_last=self._setting["test"]["drop_last"],
            num_workers=self._setting["test"]["num_workers"],
            pin_memory=True,
        )

        return test

    def _make_save_dir(self) -> Optional[str]:
        self.log.info(f'Saving tuner in {self._setting["save_dir"]}')
        save_dir = os.path.join(
            self._config["io"]["data_dir"],
            self._setting["save_dir"],
        )
        try:
            os.makedirs(save_dir, exist_ok=False)
        except FileExistsError:
            return None

        # dump configuration file
        with open(os.path.join(save_dir, "endure.toml"), "w") as fid:
            toml.dump(self._config, fid)

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
        disable_tqdm = self.log.level == logging.DEBUG

        trainer = Trainer(
            log=self.log,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            test_data=test_data,
            scheduler=scheduler,
            max_epochs=self._setting["max_epochs"],
            use_gpu_if_avail=self._setting["use_gpu_if_avail"],
            base_dir=model_base_dir,
            disable_tqdm=disable_tqdm,
            no_checkpoint=self._setting["no_checkpoint"],
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
