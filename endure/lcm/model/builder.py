from typing import Any, Optional

from torch import nn
import logging
import torch

from endure.lcm.model import FlexibleModel, QModel, QIntModel, ClassicModel
from endure.lcm.model.qlsm_model import QModelRefactor


class LearnedCostModelBuilder:
    def __init__(self, config: dict[str, Any]):
        self.log = logging.getLogger(self._config["log"]["name"])
        self._config = config
        self._models = {
            "KLSM": FlexibleModel,
            "QLSM": QModel,
            "QLSMIntegerVars": QIntModel,
            "Level": ClassicModel,
            "Tier": ClassicModel,
            "Classic": ClassicModel,
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

    def _build_qmodel(self):
        num_feats = len(self._config["lcm"]["input_features"])
        cap_range = (
            self._config["lsm"]["size_ratio"]["max"] -
            self._config["lsm"]["size_ratio"]["min"] + 1
        )
        embedding_size = self._config["lcm"]["embedding_size"]
        hidden_length = self._config["lcm"]["hidden_length"]
        hidden_width = self._config["lcm"]["hidden_width"]
        dropout = self._config["lcm"]["dropout"]
        norm_layer = None
        out_width = len(self._config["lcm"]["output_features"])
        if self._config["lcm"]["norm_layer"] == "Batch":
            norm_layer = nn.BatchNorm1d
        elif self._config["lcm"]["norm_layer"] == "Layer":
            norm_layer = nn.LayerNorm

        model = QModelRefactor(
            num_feats=num_feats,
            capacity_range=cap_range,
            embedding_size=embedding_size,
            hidden_length=hidden_length,
            hidden_width=hidden_width,
            dropout_percentage=dropout,
            out_width=out_width,
            norm_layer=norm_layer,
        )

        return model
