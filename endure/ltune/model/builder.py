import torch
from typing import Tuple
from torch import nn
from endure.lsm.types import Policy

from endure.ltune.model import ClassicTuner, QLSMTuner, KapLSMTuner
from endure.ltune.data.input_features import kINPUT_FEATS


class LTuneModelBuilder:
    def __init__(
        self,
        hidden_length: int = 1,
        hidden_width: int = 64,
        norm_layer: str = "Batch",
        dropout: float = 0.0,
        categorical_mode: str = "gumbel",
        size_ratio_range: Tuple[int, int] = (2, 31),
        max_levels: int = 16,
    ) -> None:
        self.hidden_length = hidden_length
        self.hidden_width = hidden_width
        self.dropout = dropout
        self.categorical_mode = categorical_mode
        self.max_levels = max_levels
        self.size_ratio_min, self.size_ratio_max = size_ratio_range
        self.capacity_range = self.size_ratio_max - self.size_ratio_min + 1

        self.norm_layer = nn.BatchNorm1d
        if norm_layer == "Layer":
            self.norm_layer = nn.LayerNorm

        self._models = {
            Policy.Classic: ClassicTuner,
            Policy.QFixed: QLSMTuner,
            Policy.KHybrid: KapLSMTuner,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, policy: Policy) -> torch.nn.Module:
        feat_list = kINPUT_FEATS

        kwargs = {
            "num_feats": len(feat_list),
            "capacity_range": self.capacity_range,
            "hidden_length": self.hidden_length,
            "hidden_width": self.hidden_width,
            "dropout_percentage": self.dropout,
            "norm_layer": self.norm_layer,
        }

        model_class = self._models.get(policy, None)
        if model_class is None:
            raise NotImplementedError("Tuner for LSM Design not implemented.")

        if model_class is KapLSMTuner:
            kwargs["num_kap"] = self.max_levels
            kwargs["categorical_mode"] = self.categorical_mode

        model = model_class(**kwargs)

        return model
