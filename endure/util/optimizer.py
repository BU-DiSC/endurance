import logging
from typing import Any
import torch.optim as Opt

from torch.nn import Module


class OptimizerBuilder:
    def __init__(self, config: dict[str, Any]):
        self.log = logging.getLogger(config["log"]["name"])
        self._config = config

    def _build_adam(self, model: Module) -> Opt.Adam:
        return Opt.Adam(
            model.parameters(),
            lr=self._config["train"]["optimizer"]["Adam"]["lr"],
        )

    def _build_adagrad(self, model: Module) -> Opt.Adagrad:
        return Opt.Adagrad(
            model.parameters(),
            lr=self._config["train"]["optimizer"]["Adagrad"]["lr"],
        )

    def _build_sgd(self, model: Module) -> Opt.SGD:
        return Opt.SGD(
            model.parameters(),
            lr=self._config["train"]["optimizer"]["SGD"]["lr"],
        )

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

        opt_builder = optimizers.get(optimizer_choice, None)
        if opt_builder is None:
            self.log.warn("Invalid optimizer choice, defaulting to SGD")
            return self._build_sgd(model)
        optimizer = opt_builder(model)

        return optimizer
