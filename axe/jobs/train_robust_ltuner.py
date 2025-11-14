#!/usr/bin/env python
import csv
import logging
import os
from typing import Optional, Tuple

import polars as pl
import toml
import torch
import typer
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing_extensions import Annotated

from axe.lsm.types import LSMBounds, Policy
from axe.ltuner.data.schema import LTunerDataSchema
from axe.ltuner.model.builder import LTuneModelBuilder
from axe.ltuner.robust_loss import LearnedRobustLoss
from axe.util.lr_scheduler import LRSchedulerBuilder
from axe.util.optimizer import OptimizerBuilder


class TrainRobustLTuner:
    def __init__(self, cfg: dict) -> None:
        torch.autograd.set_detect_anomaly(True)
        self.log: logging.Logger = logging.getLogger(cfg["app"]["name"])
        self.disable_tqdm: bool = cfg["app"]["disable_tqdm"]
        self.use_gpu = cfg["job"]["use_gpu_if_avail"]
        self.device = torch.device("cpu")
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.policy: Policy = getattr(Policy, cfg["lsm"]["policy"])
        self.bounds: LSMBounds = LSMBounds(**cfg["lsm"]["bounds"])
        self.schema = LTunerDataSchema(self.policy, self.bounds, robust=True)
        self.jcfg = cfg["job"]["train_ltuner"]
        self.cfg = cfg

        # Build everything we need for training
        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        torch.set_float32_matmul_precision("high")
        self.training_data, self.validate_data = self._build_data()
        self.train_kwargs = cfg["ltuner"]["train_kwargs"]
        self.validate_kwargs = cfg["ltuner"]["validate_kwargs"]

    def _build_loss_fn(self) -> torch.nn.Module:
        loss = LearnedRobustLoss(self.cfg, self.jcfg["loss_fn_path"]).to(self.device)

        return loss

    def _build_model(self) -> torch.nn.Module:
        model = LTuneModelBuilder(
            schema=self.schema, **self.cfg["ltuner"]["model"]
        ).build(robust=True)
        model.to(self.device)

        return model

    def _build_optimizer(self, model) -> torch.optim.Optimizer:
        return OptimizerBuilder(self.cfg["optimizer"]).build(
            optimizer_choice=self.jcfg["optimizer"], model=model
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        return LRSchedulerBuilder(self.cfg["scheduler"]).build(
            optimizer, self.jcfg["lr_scheduler"]
        )

    def _build_data(self) -> Tuple[DataLoader, DataLoader]:
        table = pl.read_parquet(self.jcfg["data_dir"])
        dataset = table.to_torch(
            return_type="dataset",
            features=self.schema.feat_cols(),
            label=self.schema.label_cols(),
            dtype=pl.Float32,
        )
        train_len = int(len(dataset) * self.jcfg["data_split"])
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        training_data = DataLoader(
            dataset=train_set,
            batch_size=self.jcfg["batch_size"],
            num_workers=self.jcfg["num_workers"],
            shuffle=True,
        )
        validate_data = DataLoader(
            dataset=val_set,
            batch_size=8 * self.jcfg["batch_size"],
            num_workers=self.jcfg["num_workers"],
        )

        return training_data, validate_data

    def _make_save_dir(self) -> None:
        self.log.info(f"Saving tuner in {self.jcfg['save_dir']}")
        os.makedirs(self.jcfg["save_dir"], exist_ok=False)
        if not self.jcfg["no_checkpoint"]:
            os.makedirs(os.path.join(self.jcfg["save_dir"], "checkpoints"))
        with open(os.path.join(self.jcfg["save_dir"], "axe.toml"), "w") as fid:
            toml.dump(self.cfg, fid)

    def temp_step(self, decay_rate: float = 0.97, floor: float = 0.5):
        self.train_kwargs["temp"] *= decay_rate
        if self.train_kwargs["temp"] < floor:
            self.train_kwargs["temp"] = floor

    def train_step(self, feats: Tensor, labels: Tensor, **kwargs) -> float:
        label = labels.to(self.device)
        feats = feats.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(feats, **kwargs)
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_loop(self) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.training_data, ncols=80, disable=self.disable_tqdm)
        for batch, (feats, labels) in enumerate(pbar):
            loss = self.train_step(feats, labels, **self.train_kwargs)
            if batch % (25) == 0:
                pbar.set_description(f"training loss {loss:e}")
            total_loss += loss
            if self.scheduler is not None:
                self.scheduler.step()
            self.temp_step()

        return total_loss / len(self.training_data)

    def validate_step(self, feats: Tensor, labels: Tensor, **kwargs) -> float:
        with torch.no_grad():
            labels = labels.to(self.device)
            feats = feats.to(self.device)
            pred = self.model(feats, **kwargs)
            validate_loss = self.loss_fn(pred, labels).item()

        return validate_loss

    def validate_loop(self) -> float:
        self.model.eval()
        test_loss = 0
        pbar = tqdm(self.validate_data, ncols=80, disable=self.disable_tqdm)
        for feats, labels in pbar:
            loss = self.validate_step(feats, labels, **self.validate_kwargs)
            pbar.set_description(f"validate loss {loss:e}")
            test_loss += loss

        return test_loss / len(self.validate_data)

    def save_model(self, fname: str, **kwargs) -> None:
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        save_dict.update(kwargs)
        torch.save(save_dict, os.path.join(self.jcfg["save_dir"], fname))

    def run(self):
        self.log.info("[Job] Training LCM")
        self._make_save_dir()

        loss_file = os.path.join(self.jcfg["save_dir"], "losses.csv")
        loss_min = self.validate_loop()
        with open(loss_file, "w") as fid:
            loss_csv_write = csv.writer(fid)
            loss_csv_write.writerow(["epoch", "train_loss", "test_loss"])
            loss_csv_write.writerow([0, loss_min, loss_min])

        max_epochs = self.jcfg["max_epochs"]
        for epoch in range(max_epochs):
            self.log.info(f"Epoch: [{epoch+1}/{max_epochs}]")
            train_loss = self.train_loop()
            curr_loss = self.validate_loop()
            self.log.info(f"Training loss: {train_loss:.5e}")
            self.log.info(f"Validate loss: {curr_loss:.5e}")
            if not self.jcfg["no_checkpoint"]:
                self.save_model(f"checkpoints/epoch{epoch:02d}.model", loss=curr_loss)

            if curr_loss < loss_min:
                loss_min = curr_loss
                self.log.info("New minmum loss saving best model")
                self.save_model("best_model.model", loss=loss_min, epoch=epoch)
            with open(loss_file, "a") as fid:
                write = csv.writer(fid)
                write.writerow([epoch + 1, train_loss, curr_loss])

        self.log.info("Training finished")


def train_robust_ltuner(
    ctx: typer.Context,
    max_epochs: Annotated[
        int, typer.Option("--max-epochs", help="Maximum number of training epochs.")
    ],
    save_dir: Annotated[
        str, typer.Option("--save-dir", help="Directory to save the trained model.")
    ],
    loss_fn_path: Annotated[
        str,
        typer.Option(
            "--loss-fn-path", help="Path to the learned cost model for the loss function."
        ),
    ],
    optimizer: Annotated[
        str, typer.Option("--optimizer", help="Optimizer to use for training.")
    ],
    lr_scheduler: Annotated[
        str, typer.Option("--lr-scheduler", help="Learning rate scheduler to use.")
    ],
    no_checkpoint: Annotated[
        bool, typer.Option("--no-checkpoint", help="Disable model checkpoints.")
    ],
    data_split: Annotated[
        float, typer.Option("--data-split", help="Train/validation data split ratio.")
    ],
    data_dir: Annotated[
        str, typer.Option("--data-dir", help="Directory containing the training data.")
    ],
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Batch size for training.")
    ],
    shuffle: Annotated[bool, typer.Option("--shuffle", help="Shuffle the training data.")],
    num_workers: Annotated[
        int,
        typer.Option(
            "--num-workers", help="Number of worker processes for data loading."
        ),
    ],
    policy: Annotated[
        str, typer.Option("--lsm-policy", help="LSM policy to use.")
    ],
    use_gpu_if_avail: Annotated[
        bool, typer.Option("--use-gpu-if-avail", help="Use GPU if available.")
    ],
):
    """Train the Robust Learned Tuner (LTuner)."""
    config = ctx.obj
    job_config = config["job"]["train_ltuner"]

    if max_epochs is not None:
        job_config["max_epochs"] = max_epochs
    if save_dir is not None:
        job_config["save_dir"] = save_dir
    if loss_fn_path is not None:
        job_config["loss_fn_path"] = loss_fn_path
    if optimizer is not None:
        job_config["optimizer"] = optimizer
    if lr_scheduler is not None:
        job_config["lr_scheduler"] = lr_scheduler
    if no_checkpoint:
        job_config["no_checkpoint"] = no_checkpoint
    if data_split is not None:
        job_config["data_split"] = data_split
    if data_dir is not None:
        job_config["data_dir"] = data_dir
    if batch_size is not None:
        job_config["batch_size"] = batch_size
    if shuffle:
        job_config["shuffle"] = shuffle
    if num_workers is not None:
        job_config["num_workers"] = num_workers
    if policy is not None:
        config["lsm"]["policy"] = policy
    if use_gpu_if_avail:
        config["job"]["use_gpu_if_avail"] = use_gpu_if_avail

    TrainRobustLTuner(config).run()