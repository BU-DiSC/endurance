from typing import Any

from torch import nn
import logging
import torch

from endure.lcm.data.input_features import kINPUT_FEATS_DICT
from endure.lcm.model import KapModel, QModel, ClassicModel
from endure.lsm.types import Policy


class LearnedCostModelBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.log = logging.getLogger(config["log"]["name"])
        self._config = config
        self._models = {
            Policy.KHybrid: KapModel,
            Policy.QFixed: QModel,
            Policy.Classic: ClassicModel,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, policy: Policy) -> torch.nn.Module:
        feats_list = kINPUT_FEATS_DICT.get(policy, None)
        if feats_list is None:
            raise TypeError("Illegal policy")

        num_feats = len(feats_list)
        max_levels = self._config["lsm"]["max_levels"]
        if "K" in feats_list:
            # Add number of features to expand K to K0, K1, ..., K_maxlevels
            num_feats += max_levels - 1
        capacity_range = (
            self._config["lsm"]["size_ratio"]["max"]
            - self._config["lsm"]["size_ratio"]["min"]
            + 1
        )

        model_params = self._config["lcm"]["model"]
        args = {
            "num_feats": num_feats,
            "capacity_range": capacity_range,
            "embedding_size": model_params["embedding_size"],
            "hidden_length": model_params["hidden_length"],
            "hidden_width": model_params["hidden_width"],
            "dropout_percentage": model_params["dropout"],
            "decision_dim": model_params["decision_dim"],
        }

        if model_params["norm_layer"] == "Batch":
            args["norm_layer"] = nn.BatchNorm1d
        elif model_params["norm_layer"] == "Layer":
            args["norm_layer"] = nn.LayerNorm

        model_class = self._models.get(policy, None)
        if model_class is None:
            raise NotImplementedError(f"Model for LSM Design not implemented yet")

        if model_class is ClassicModel:
            args["policy_embedding_size"] = model_params["policy_embedding_size"]

        if model_class is KapModel:
            args["max_levels"] = max_levels

        model = model_class(**args)

        return model
