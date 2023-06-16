from typing import Any, Optional
import torch
import logging

from endure.lcm.model.flexible_model import FlexibleModel
from endure.lcm.model.classic_model import ClassicModel
from endure.lcm.model.qlsm_model import QModel


class LearnedCostModelBuilder:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self.log = logging.getLogger(self._config["log"]["name"])
        self._models = {
            "KLSM": FlexibleModel,
            "QLSM": QModel,
            "Level": ClassicModel,
            "Tier": ClassicModel,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: Optional[str] = None) -> torch.nn.Module:
        lsm_design: str = self._config["lsm"]["design"]
        if choice is None:
            choice = lsm_design

        model = self._models.get(choice, None)
        if model is None:
            self.log.warn("Invalid model architecture. Defaulting to classic")
            return ClassicModel(self._config)
        model = model(self._config)

        return model
