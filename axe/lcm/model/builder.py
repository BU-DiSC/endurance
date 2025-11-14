from torch import nn
import torch

from axe.config import LCMConfig
from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model import KapLCM, QHybridLCM, ClassicLCM, FluidLCM
from axe.lsm.types import Policy


class LearnedCostModelBuilder:
    def __init__(
        self,
        config: LCMConfig,
        schema: LCMDataSchema,
    ) -> None:
        self.max_levels = schema.bounds.max_considered_levels
        self.capacity_range = (
            schema.bounds.size_ratio_range[1] - schema.bounds.size_ratio_range[0]
        )
        self.schema = schema
        self.config = config

        self.norm_layer = nn.BatchNorm1d
        if config.norm_layer == "Norm":
            self.norm_layer = nn.LayerNorm

        self._models = {
            Policy.Classic: ClassicLCM,
            Policy.QHybrid: QHybridLCM,
            Policy.Fluid: FluidLCM,
            Policy.Kapacity: KapLCM,
        }

    def get_choices(self):
        return self._models.keys()

    def build(self, **kwargs) -> torch.nn.Module:
        feat_columns = self.schema.feat_cols()
        num_feats = len(feat_columns)
        args = {
            "num_feats": num_feats,
            "capacity_range": self.capacity_range,
            "embedding_size": self.config.embedding_size,
            "hidden_length": self.config.hidden_length,
            "hidden_width": self.config.hidden_width,
            "dropout_percentage": self.config.dropout,
            "decision_dim": self.config.decision_dim,
            "norm_layer": self.norm_layer,
        }
        args.update(kwargs)

        model_class = self._models.get(self.schema.policy, None)
        if model_class is None:
            raise NotImplementedError(f"Policy {self.schema.policy} not Implemented")

        if model_class is ClassicLCM:
            args["policy_embedding_size"] = self.config.policy_embedding_size

        if model_class is KapLCM:
            args["max_levels"] = self.max_levels

        model = model_class(**args)

        return model
