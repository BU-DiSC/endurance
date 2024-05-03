from typing import Any
import torch.optim as Opt

from torch.nn import Module


class OptimizerBuilder:
    def __init__(self, optimizer_config: dict[str, Any]):
        self.optimizer_config = optimizer_config

    def _build_adam(self, model: Module) -> Opt.Adam:
        return Opt.Adam(model.parameters(), **self.optimizer_config["Adam"])

    def _build_adagrad(self, model: Module) -> Opt.Adagrad:
        return Opt.Adagrad(model.parameters(), **self.optimizer_config["Adagrad"])

    def _build_sgd(self, model: Module) -> Opt.SGD:
        return Opt.SGD(model.parameters(), **self.optimizer_config["SGD"])

    def build_optimizer(
        self,
        optimizer_choice: str,
        model: Module,
    ) -> Opt.Optimizer:
        optimizers = {
            "Adam": self._build_adam,
            "Adagrad": self._build_adagrad,
            "SGD": self._build_sgd,
        }

        optimizer_class = optimizers.get(optimizer_choice, None)
        if optimizer_class is None:
            raise KeyError
        optimizer = optimizer_class(model)

        return optimizer
