from typing import Any, Optional

from torch import nn
import logging
import torch

from endure.lcm.model import FlexModel, QModel, ClassicModel


class LearnedCostModelBuilder:
    def __init__(self, config: dict[str, Any]):
        self.log = logging.getLogger(config["log"]["name"])
        self._config = config
        self._models = {
            "KLSM": FlexModel,
            "QLSM": QModel,
            "Classic": ClassicModel,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: Optional[str] = None) -> torch.nn.Module:
        lsm_design: str = self._config["lsm"]["design"]
        if choice is None:
            choice = lsm_design

        num_feats = len(self._config["lcm"]["input_features"])
        capacity_range = (
            self._config["lsm"]["size_ratio"]["max"] -
            self._config["lsm"]["size_ratio"]["min"] + 1
        )
        model_params = self._config["lcm"]["model"]
        embedding_size = model_params["embedding_size"]
        policy_embedding_size = model_params["policy_embedding_size"]
        hidden_length = model_params["hidden_length"]
        hidden_width = model_params["hidden_width"]
        dropout_percentage = model_params["dropout"]
        out_width = len(self._config["lcm"]["output_features"])

        norm_layer = None
        if model_params["norm_layer"] == "Batch":
            norm_layer = nn.BatchNorm1d
        elif model_params["norm_layer"] == "Layer":
            norm_layer = nn.LayerNorm

        model_class = self._models.get(choice, None)
        if model_class is None:
            self.log.warn("Invalid model architecture. Defaulting to classic")
            model_class = ClassicModel
        if model_class == ClassicModel:
            model = model_class(
                num_feats=num_feats,
                capacity_range=capacity_range,
                embedding_size=embedding_size,
                hidden_length=hidden_length,
                hidden_width=hidden_width,
                dropout_percentage=dropout_percentage,
                out_width=out_width,
                norm_layer=norm_layer,
                policy_embedding_size=policy_embedding_size,
            )
        else:
            model = model_class(
                num_feats=num_feats,
                capacity_range=capacity_range,
                embedding_size=embedding_size,
                hidden_length=hidden_length,
                hidden_width=hidden_width,
                dropout_percentage=dropout_percentage,
                out_width=out_width,
                norm_layer=norm_layer,
            )

        return model

