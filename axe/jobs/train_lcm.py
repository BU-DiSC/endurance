#!/usr/bin/env python
import csv
import logging
import os
from typing import Literal, Optional, Tuple

import polars as pl
import toml
import torch
import typer
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing_extensions import Annotated

from axe.config import AxeConfig
from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model.builder import LearnedCostModelBuilder
from axe.lsm.types import LSMBounds, Policy
from axe.util.losses import LossBuilder
from axe.util.lr_scheduler import LRSchedulerBuilder
from axe.util.optimizer import OptimizerBuilder

logger = logging.getLogger(__name__)


class TrainLCM:
    def __init__(
        self,
        config: AxeConfig,
        data_dir: str,
        save_dir: str,
        *,
        max_epochs: int = 10,
        data_split: float = 0.9,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 1,
        checkpoints: bool = False,
    ) -> None:
        self.device = torch.device("cpu")
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.policy: Policy = config.lsm.policy
        self.bounds: LSMBounds = config.lsm.bounds
        self.schema = LCMDataSchema(self.policy, self.bounds)
        self.config = config
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.data_split = data_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.checkpoints = checkpoints
        self.shuffle = shuffle

        # Build everything we need for training
        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        torch.set_float32_matmul_precision("high")
        self.training_data, self.validate_data = self._build_data()

    def _build_loss_fn(self) -> torch.nn.Module:
        choice = self.config.loss.name
        loss = LossBuilder(self.config.loss).build(choice)
        logger.info(f"Loss function: {choice}")
        if loss is None:
            logger.warning(f"Invalid loss function: {choice}")
            raise KeyError
        if self.config.use_gpu and torch.cuda.is_available():
            loss.to("cuda")

        return loss

    def _build_model(self) -> torch.nn.Module:
        model = LearnedCostModelBuilder(
            config=self.config.lcm, schema=self.schema
        ).build()
        # model.compile()
        model.to(self.device)

        return model

    def _build_optimizer(self, model) -> torch.optim.Optimizer:
        return OptimizerBuilder(self.config.optimizer).build(
            optimizer_choice=self.config.optimizer.name, model=model
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        return LRSchedulerBuilder(self.config.scheduler).build(
            optimizer, self.config.scheduler.name
        )

    def _build_data(self) -> Tuple[DataLoader, DataLoader]:
        table = self.schema.read_data(self.data_dir, preprocess=True)
        dataset = table.to_torch(
            return_type="dataset",
            features=self.schema.feat_cols(),
            label=self.schema.label_cols(),
            dtype=pl.Float32,
        )
        train_len = int(len(dataset) * self.data_split)
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        training_data = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        validate_data = DataLoader(
            dataset=val_set,
            batch_size=8 * self.batch_size,
            num_workers=self.num_workers,
        )

        return training_data, validate_data

    def _make_save_dir(self) -> None:
        logger.info(f"Saving tuner in {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=False)
        if self.checkpoints:
            os.makedirs(os.path.join(self.save_dir, "checkpoints"))
        with open(os.path.join(self.save_dir, "axe.toml"), "w") as fid:
            toml.dump(self.config.model_dump(), fid)

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
        pbar = tqdm(self.training_data, ncols=80, disable=self.config.disable_tqdm)
        for batch, (feats, labels) in enumerate(pbar):
            loss = self.train_step(feats, labels)
            if batch % (100) == 0:
                pbar.set_description(f"training loss {loss:e}")
            total_loss += loss
            if self.scheduler is not None:
                self.scheduler.step()

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
        validate_loss = 0
        pbar = tqdm(self.validate_data, ncols=80, disable=self.config.disable_tqdm)
        for feats, labels in pbar:
            loss = self.validate_step(feats, labels)
            pbar.set_description(f"validate loss {loss:e}")
            validate_loss += loss

        return validate_loss / len(self.validate_data)

    def save_model(self, fname: str, **kwargs) -> None:
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        save_dict.update(kwargs)
        torch.save(save_dict, os.path.join(self.save_dir, fname))

    def run(self):
        logger.info("[Job] Training LCM")
        self._make_save_dir()

        loss_file = os.path.join(self.save_dir, "losses.csv")
        with open(loss_file, "w") as fid:
            loss_csv_write = csv.writer(fid)
            loss_csv_write.writerow(["epoch", "train_loss", "test_loss"])

        max_epochs = self.max_epochs
        loss_min = self.validate_loop()
        with open(loss_file, "a") as fid:
            write = csv.writer(fid)
            write.writerow([0, loss_min, loss_min])
        for epoch in range(max_epochs):
            logger.info(f"Epoch: [{epoch + 1}/{max_epochs}]")
            train_loss = self.train_loop()
            curr_loss = self.validate_loop()
            logger.info(f"Training loss: {train_loss}")
            logger.info(f"Validate loss: {curr_loss}")
            if self.checkpoints:
                self.save_model(f"checkpoints/epoch{epoch:02d}.model", loss=curr_loss)

            if curr_loss < loss_min:
                loss_min = curr_loss
                logger.info("New minmum loss saving best model")
                self.save_model("best_model.model", loss=loss_min, epoch=epoch)
            with open(loss_file, "a") as fid:
                write = csv.writer(fid)
                write.writerow([epoch + 1, train_loss, curr_loss])

        logger.info("Training finished")


def train_lcm(
    ctx: typer.Context,
    save_dir: Annotated[
        str, typer.Option("--save-dir", help="Directory to save the trained model.")
    ],
    data_dir: Annotated[
        str, typer.Option("--data-dir", help="Directory containing the training data.")
    ],
    max_epochs: Annotated[
        int, typer.Option("--max-epochs", help="Maximum number of training epochs.")
    ] = 10,
    data_split: Annotated[
        float, typer.Option("--data-split", help="Train/validation data split ratio.")
    ] = 0.9,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Batch size for training.")
    ] = 64,
    shuffle: Annotated[
        bool,
        typer.Option(
            "--shuffle/--no-shuffle",
            is_flag=True,
            help="Shuffle the training data.",
        ),
    ] = True,
    checkpoints: Annotated[
        bool,
        typer.Option(
            "--checkpoints/--no-checkpoints",
            is_flag=True,
            help="Disable model checkpoints.",
        ),
    ] = True,
    num_workers: Annotated[
        int,
        typer.Option(
            "--num-workers", help="Number of worker processes for data loading."
        ),
    ] = 1,
    loss_fn: Annotated[
        str, typer.Option("--loss-fn", help="Loss function to use for training.")
    ] = "MSE",
    optimizer: Annotated[
        Literal["Adam", "AdamW", "SGD", "Adagrad"] | None,
        typer.Option("--optimizer", help="Optimizer to use for training."),
    ] = None,
    lr_scheduler: Annotated[
        Literal["CosineAnnealingLR", "Exponential", "Constant"] | None,
        typer.Option("--lr-scheduler", help="Learning rate scheduler to use."),
    ] = None,
    policy: Annotated[
        str | None, typer.Option("--lsm-policy", help="LSM policy to use.")
    ] = None,
    use_gpu: Annotated[
        bool | None,
        typer.Option(
            "--use-gpu/--no-use-gpu", is_flag=True, help="Use GPU if available."
        ),
    ] = None,
):
    """Train the Learned Cost Model (LCM)."""
    config: AxeConfig = ctx.obj

    if use_gpu:
        config.use_gpu = True
    if policy is not None:
        config.lsm.policy = getattr(Policy, policy)
    if loss_fn is not None:
        config.loss.name = loss_fn
    if optimizer is not None:
        config.optimizer.name = optimizer
    if lr_scheduler is not None:
        config.scheduler.name = lr_scheduler

    TrainLCM(
        config,
        data_dir=data_dir,
        save_dir=save_dir,
        max_epochs=max_epochs,
        batch_size=batch_size,
        data_split=data_split,
        num_workers=num_workers,
        shuffle=shuffle,
        checkpoints=checkpoints,
    ).run()
