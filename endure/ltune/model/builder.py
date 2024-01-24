import torch
import logging
from typing import Any, Optional
from torch import nn
from reinmax import reinmax

from endure.ltune.model import ClassicTuner, QLSMTuner, KapLSMTuner


class LTuneModelBuilder:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self.log = logging.getLogger(self._config["log"]["name"])
        self._models = {
            # "Tier": ClassicTuner,
            # "Level": ClassicTuner,
            "KLSM": KapLSMTuner,
            "Classic": ClassicTuner,
            "QLSM": QLSMTuner,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: Optional[str] = None) -> torch.nn.Module:
        lsm_design: str = self._config["lsm"]["design"]
        if choice is None:
            choice = lsm_design

        model_params = self._config["ltune"]["model"]
        capacity_range = (
            self._config["lsm"]["size_ratio"]["max"] -
            self._config["lsm"]["size_ratio"]["min"] + 1
        )
        args = {
            'num_feats': len(self._config["ltune"]["input_features"]),
            'capacity_range': capacity_range,
            'hidden_length': model_params["hidden_length"],
            'hidden_width': model_params["hidden_width"],
            'dropout_percentage': model_params["dropout"],
        }

        if model_params["norm_layer"] == "Batch":
            args['norm_layer'] = nn.BatchNorm1d
        elif model_params["norm_layer"] == "Layer":
            args['norm_layer'] = nn.LayerNorm

        model_class = self._models.get(choice, None)
        if model_class is None:
            raise NotImplementedError(f"Model for LSM Design not implemented yet")

        if model_class is KapLSMTuner:
            args['num_kap'] = self._config['lsm']['max_levels']
            args['categorical_mode'] = model_params.get('categorical_mode', 'gumbel')

        model = model_class(**args)

        return model
