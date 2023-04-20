import logging
import torch.optim as Opt

from torch.nn import Module


class LRSchedulerBuilder:
    def __init__(self, config: dict[str, ...]):
        self.log = logging.getLogger(config["log"]["name"])
        self._config = config

    def _build_cosine_anneal(
        self,
        optimizer: Opt.Optimizer,
    ) -> Opt.lr_scheduler.CosineAnnealingLR:
        return Opt.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **self._config["train"]["scheduler"]["CosineAnnealingLR"],
        )

    def _build_exponential(
        self,
        optimizer: Opt.Optimizer,
    ) -> Opt.lr_scheduler.ExponentialLR:
        return Opt.lr_scheduler.ExponentialLR(
            optimizer,
            **self._config["train"]["scheduler"]["Exponential"],
        )

    def build_scheduler(
        self,
        optimizer: Opt.Optimizer,
        scheduler_choice: str = "Constant",
    ) -> Opt.lr_scheduler._LRScheduler:
        schedules = {
            "CosineAnnealing": self._build_cosine_anneal,
            "Exponential": self._build_exponential,
            "Constant": None,
            "None": None,
        }
        schedule_builder = schedules.get(scheduler_choice, -1)
        if schedule_builder == -1:
            self.log.warn("Invalid scheduler, defaulting to Constant")
            return None
        if schedule_builder is None:  # Constant/None case
            return None
        scheduler = schedule_builder(optimizer)

        return scheduler
