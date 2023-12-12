#!/usr/bin/env python
import os
import torch
import logging
import toml
from typing import Any, Callable, Optional

import torch.optim as Opt
from torch.utils.data import DataLoader

from endure.ltune.data.dataset import LTuneIterableDataSet
from endure.ltune.loss import LearnedCostModelLoss
from endure.ltune.model.builder import LTuneModelBuilder
from endure.util.lr_scheduler import LRSchedulerBuilder
from endure.util.optimizer import OptimizerBuilder
from endure.util.trainer import Trainer


class LTuneTrainJob:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._setting = config["job"]["LTuneTrain"]
        self.log = logging.getLogger(self._config["log"]["name"])
        self.log.info("Running Training Job")

    def _build_loss_fn(self) -> torch.nn.Module:
        model = LearnedCostModelLoss(self._config, self._setting["loss_fn_path"])
        if self._setting["use_gpu_if_avail"] and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_model(self) -> torch.nn.Module:
        model = LTuneModelBuilder(self._config).build_model()
        if self._setting["use_gpu_if_avail"] and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_optimizer(self, model) -> Opt.Optimizer:
        builder = OptimizerBuilder(self._config)
        choice = self._setting["optimizer"]

        return builder.build_optimizer(choice, model)

    def _build_scheduler(
        self, optimizer: Opt.Optimizer
    ) -> Optional[Opt.lr_scheduler._LRScheduler]:
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
            prefetch_factor=10,
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

    def _make_save_dir(self) -> Optional[str]:
        self.log.info(f'Saving model in: {self._setting["save_dir"]}')
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

    @staticmethod
    def gumbel_temp_schedule(
        train_kwargs: dict,
        decay_rate: float = 0.95,
        floor: float = 0.01,
    ) -> None:
        train_kwargs["temp"] *= decay_rate
        if train_kwargs["temp"] < floor:
            train_kwargs["temp"] = floor

        return

    @staticmethod
    def reinmax_temp_schedule(
        train_kwargs: dict,
        decay_rate: float = 0.9,
        floor: float = 1,
    ) -> None:
        train_kwargs["temp"] *= decay_rate
        if train_kwargs["temp"] < floor:
            train_kwargs["temp"] = floor

        return

    def get_train_callback(self) -> Optional[Callable[[dict], None]]:
        if not self._config["lsm"]["design"] == "KLSM":
            return None
        if self._config["ltune"]["model"]["categorical_mode"] == "reinmax":
            return lambda train_kwargs: self.reinmax_temp_schedule(train_kwargs)

        return lambda train_kwargs: self.gumbel_temp_schedule(train_kwargs)

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
        disable_tqdm = self._config["log"]["disable_tqdm"]
        callback = self.get_train_callback()

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
            no_checkpoint=self._config["job"]["LTuneTrain"]["no_checkpoint"],
            train_callback=callback
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
