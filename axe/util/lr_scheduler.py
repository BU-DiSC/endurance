from typing import Optional

import torch.optim as Opt

from axe.config import SchedulerConfig


class LRSchedulerBuilder:
    def __init__(self, config: SchedulerConfig):
        self.config = config

    def _build_cosine_anneal(
        self,
        optimizer: Opt.Optimizer,
    ) -> Opt.lr_scheduler.CosineAnnealingLR:
        return Opt.lr_scheduler.CosineAnnealingLR(optimizer, **self.config.options)

    def _build_exponential(
        self,
        optimizer: Opt.Optimizer,
    ) -> Opt.lr_scheduler.ExponentialLR:
        return Opt.lr_scheduler.ExponentialLR(optimizer, **self.config.options)

    def build(
        self, optimizer: Opt.Optimizer, scheduler_choice: str = "Constant"
    ) -> Optional[Opt.lr_scheduler._LRScheduler]:
        schedules = {
            "CosineAnnealing": self._build_cosine_anneal,
            "Exponential": self._build_exponential,
            "Constant": "Constant",
            "None": "Constant",
        }
        schedule_builder = schedules.get(scheduler_choice, "Constant")
        if schedule_builder == "Constant":
            return None
        scheduler = schedule_builder(optimizer)

        return scheduler
