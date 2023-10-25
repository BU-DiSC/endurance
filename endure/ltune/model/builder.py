import torch
import logging
from typing import Any, Optional
from torch import nn

from endure.ltune.model import ClassicTuner, QLSMTuner


class LTuneModelBuilder:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self.log = logging.getLogger(self._config["log"]["name"])
        self._models = {
            # "Tier": ClassicTuner,
            # "Level": ClassicTuner,
            "Classic": ClassicTuner,
            "QLSM": QLSMTuner,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: Optional[str] = None) -> torch.nn.Module:
        lsm_design: str = self._config["lsm"]["design"]
        if choice is None:
            choice = lsm_design

        num_feats = len(self._config["ltune"]["input_features"])
        capacity_range = (
            self._config["lsm"]["size_ratio"]["max"] -
            self._config["lsm"]["size_ratio"]["min"] + 1
        )
        model_params = self._config["ltune"]["model"]
        hidden_length = model_params["hidden_length"]
        hidden_width = model_params["hidden_width"]
        dropout_percentage = model_params["dropout"]

        norm_layer = None
        if model_params["norm_layer"] == "Batch":
            norm_layer = nn.BatchNorm1d
        elif model_params["norm_layer"] == "Layer":
            norm_layer = nn.LayerNorm

        model_class = self._models.get(choice, None)
        if model_class is None:
            self.log.warn("Invalid model architecture. Defaulting to classic")
            model_class = ClassicTuner

        model = model_class(
            num_feats=num_feats,
            capacity_range=capacity_range,
            hidden_length=hidden_length,
            hidden_width=hidden_width,
            dropout_percentage=dropout_percentage,
            norm_layer=norm_layer,
        )

        return model
