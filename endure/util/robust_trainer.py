import csv
import logging
import os
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Any, Callable, Iterable, Optional, Union
from endure.util.trainer import Trainer


class RobustTrainer(Trainer):
    def __init__(
        self,
        log: logging.Logger,
        model: Union[torch.nn.Module, Callable],
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[torch.nn.Module, Callable],
        train_data: Iterable[Union[DataLoader, Dataset]],
        test_data: Iterable[Union[DataLoader, Dataset]],
        epislon: float = 0.1,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_epochs: int = 10,
        base_dir: str = "./",
        use_gpu_if_avail: bool = False,
        model_train_kwargs: dict[str, Any] = {},
        model_test_kwargs: dict[str, Any] = {},
        disable_tqdm: bool = False,
        no_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            log=log,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            test_data=test_data,
            scheduler=scheduler,
            max_epochs=max_epochs,
            base_dir=base_dir,
            use_gpu_if_avail=use_gpu_if_avail,
            model_train_kwargs=model_train_kwargs,
            model_test_kwargs=model_test_kwargs,
            disable_tqdm=disable_tqdm,
            no_checkpoint=no_checkpoint,
        )
        self.epsilon = epislon

    def _train_step(
        self,
        label: torch.Tensor,
        features: torch.Tensor,
    ) -> float:
        label = label.to(self.device)
        features = features.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(features, **self.model_train_kwargs)
        loss = self.loss_fn(pred, label)
        loss.backward()

        assert features.grad is not None
        perturbed_input = features + (self.epsilon * features.grad.data)
        # TODO make sure input is VALID
        perturbed_pred = self.model(perturbed_input, **self.model_train_kwargs)
        perturbed_loss = self.loss_fn(perturbed_pred, label)
        self.optimizer.zero_grad()
        perturbed_loss.backward()
        self.optimizer.step()

        return loss.item()

    # def run(self) -> None:
    #     os.makedirs(self.checkpoint_dir, exist_ok=True)
    #
    #     with open(os.path.join(self.base_dir, "losses.csv"), "w") as fid:
    #         loss_csv_write = csv.writer(fid)
    #         loss_csv_write.writerow(["epoch", "train_loss", "test_loss"])
    #
    #     self.log.info("Initial test with random network")
    #     loss_min = self._test_loop()
    #     with open(os.path.join(self.base_dir, "losses.csv"), "a") as fid:
    #         write = csv.writer(fid)
    #         write.writerow([0, loss_min, loss_min])
    #     self.log.info(f"Initial Loss: {loss_min}")
    #
    #     for epoch in range(self.max_epochs):
    #         self.log.info(f"Epoch: [{epoch+1}/{self.max_epochs}]")
    #         train_loss = self._train_loop()
    #         curr_loss = self._test_loop()
    #         self.log.info(f"Train loss: {train_loss}")
    #         self.log.info(f"Test loss: {curr_loss}")
    #         self._checkpoint(epoch, curr_loss)
    #
    #         if curr_loss < loss_min:
    #             loss_min = curr_loss
    #             self.log.info("New minmum loss, saving...")
    #             self._save_model("best.model")
    #         with open(os.path.join(self.base_dir, "losses.csv"), "a") as fid:
    #             write = csv.writer(fid)
    #             write.writerow([epoch + 1, train_loss, curr_loss])
    #
    #     self.log.info("Training finished")
    #
    #     return
