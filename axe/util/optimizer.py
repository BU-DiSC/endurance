from typing import Any

import torch.optim as Opt
from torch.nn import Module

from axe.config import OptimizerConfig


class OptimizerBuilder:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.opt_kwargs = config

    def _build_adam(self, model: Module) -> Opt.Adam:
        return Opt.Adam(model.parameters(), **self.config.options)

    def _build_adagrad(self, model: Module) -> Opt.Adagrad:
        return Opt.Adagrad(model.parameters(), **self.config.options)

    def _build_sgd(self, model: Module) -> Opt.SGD:
        return Opt.SGD(model.parameters(), **self.config.options)

    def _build_adamw(self, model: Module) -> Opt.AdamW:
        return Opt.AdamW(model.parameters(), **self.config.options)

    def build(self, optimizer_choice: str, model: Module) -> Opt.Optimizer:
        optimizers = {
            "Adam": self._build_adam,
            "AdamW": self._build_adamw,
            "Adagrad": self._build_adagrad,
            "SGD": self._build_sgd,
        }

        opt_builder = optimizers.get(optimizer_choice, None)
        if opt_builder is None:
            raise KeyError
        optimizer = opt_builder(model)

        return optimizer
