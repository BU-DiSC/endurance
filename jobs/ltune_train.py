#!/usr/bin/env python
import os
import torch
import logging
import toml

import torch.optim as Opt
from torch.utils.data import DataLoader

from endure.ltune.data.dataset import LTuneIterableDataSet
from endure.ltune.loss import LearnedCostModelLoss
from endure.ltune.model.builder import LTuneModelBuilder
from endure.util.lr_scheduler import LRSchedulerBuilder
from endure.util.optimizer import OptimizerBuilder
from endure.util.trainer import Trainer


class LTuneTrainJob:
    def __init__(self, config: dict[str, ...]) -> None:
        self._config = config
        self._setting = config["job"]["LTuneTrain"]
        self.log = logging.getLogger(self._config["log"]["name"])
        self.log.info("Running Training Job")

    def _build_loss_fn(self) -> torch.nn.Module:
        return LearnedCostModelLoss(self._config, self._setting["loss_fn_path"])

    def _build_model(self) -> torch.nn.Module:
        builder = LTuneModelBuilder(self._config)

        return builder.build_model()

    def _build_optimizer(self, model) -> Opt.Optimizer:
        builder = OptimizerBuilder(self._config)
        choice = self._setting["optimizer"]

        return builder.build_optimizer(choice, model)

    def _build_scheduler(
        self, optimizer: Opt.Optimizer
    ) -> Opt.lr_scheduler._LRScheduler:
        builder = LRSchedulerBuilder(self._config)
        choice = self._setting["lr_scheduler"]

        return builder.build_scheduler(optimizer, choice)

    def _build_train(self) -> DataLoader:
        train_dir = os.path.join(
            self._config["io"]["data_dir"],
            self._setting["train"]["dir"],
        )
        train_data = LTuneIterableDataSet(
            config=self._config,
            folder=train_dir,
            shuffle=self._setting["train"]["shuffle"],
            format=self._setting["train"]["format"],
        )
        train = DataLoader(
            train_data,
            batch_size=self._setting["train"]["batch_size"],
            drop_last=self._setting["train"]["drop_last"],
            num_workers=self._setting["train"]["num_workers"],
            pin_memory=True,
            prefetch_factor=50,
        )

        return train

    def _build_test(self) -> DataLoader:
        test_dir = os.path.join(
            self._config["io"]["data_dir"],
            self._setting["test"]["dir"],
        )
        test_data = LTuneIterableDataSet(
            config=self._config,
            folder=test_dir,
            shuffle=self._setting["test"]["shuffle"],
            format=self._setting["test"]["format"],
        )
        test = DataLoader(
            test_data,
            batch_size=self._setting["test"]["batch_size"],
            drop_last=self._setting["test"]["drop_last"],
            num_workers=self._setting["test"]["num_workers"],
        )

        return test

    def _dumpconfig(self, save_dir: str) -> None:
        with open(os.path.join(save_dir, "endure.toml"), "w") as fid:
            toml.dump(self._config, fid)

        return

    def _make_save_dir(self) -> str:
        self.log.info(f'Saving model in: {self._setting["save_dir"]}')
        save_dir = os.path.join(
            self._config["io"]["data_dir"],
            self._setting["save_dir"],
        )
        os.makedirs(save_dir, exist_ok=True)
        self._dumpconfig(save_dir)

        return save_dir

    def run(self) -> Trainer:
        model_base_dir = self._make_save_dir()

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
            model_train_kwargs=self._config["ltune"]["model"]["train_kwargs"],
            model_test_kwargs=self._config["ltune"]["model"]["test_kwargs"],
            disable_tqdm=disable_tqdm,
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

    a = LTuneTrainJob(config)
    a.run()
