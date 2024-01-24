from typing import Any, Optional

from torch import nn
import logging
import torch

from endure.lcm.model import KapModel, QModel, ClassicModel


class LearnedCostModelBuilder:
    def __init__(self, config: dict[str, Any]):
        self.log = logging.getLogger(config["log"]["name"])
        self._config = config
        self._models = {
            "KHybrid": KapModel,
            "QFixed": QModel,
            "Classic": ClassicModel,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: Optional[str] = None) -> torch.nn.Module:
        lsm_design: str = self._config["lsm"]["design"]
        if choice is None or choice == "Auto":
            choice = lsm_design

        max_levels = self._config["lsm"]["max_levels"]
        num_feats = len(self._config["lcm"]["input_features"])
        if "K" in self._config["lcm"]["input_features"]:
            # Add number of features to expand K to K0, K1, ..., K_maxlevels
            num_feats += max_levels - 1
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
                norm_layer=norm_layer,
                policy_embedding_size=policy_embedding_size,
            )
        elif model_class == KapModel:
            model = model_class(
                num_feats=num_feats,
                capacity_range=capacity_range,
                embedding_size=embedding_size,
                hidden_length=hidden_length,
                hidden_width=hidden_width,
                dropout_percentage=dropout_percentage,
                norm_layer=norm_layer,
                max_levels=max_levels,
            )
        else:
            model = model_class(
                num_feats=num_feats,
                capacity_range=capacity_range,
                embedding_size=embedding_size,
                hidden_length=hidden_length,
                hidden_width=hidden_width,
                dropout_percentage=dropout_percentage,
                norm_layer=norm_layer,
            )

        return model

