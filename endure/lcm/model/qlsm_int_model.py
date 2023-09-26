from typing import Any

from torch import nn
# import torch
# import torch.nn.functional as F


class QIntModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.max_levels = config["lsm"]["max_levels"]
        self.params = config["lcm"]["model"]["flexible"]
        self.features = config["lcm"]["input_features"]
        self.num_features = len(self.features)

        modules = []
        self.size_ratio_range = (
            config["lsm"]["size_ratio"]["max"] - config["lsm"]["size_ratio"]["min"] + 1
        )

        # Q and T are dummy vars
        in_dim = self.num_features

        if self.params["normalize"] == "Layer":
            modules.append(nn.LayerNorm(in_dim))
        elif self.params["normalize"] == "Batch":
            modules.append(nn.BatchNorm1d(in_dim))
        # else: No normalization layer

        hidden_dim = self.params["layer_size"]
        modules.append(nn.Linear(in_dim, hidden_dim))
        modules.append(nn.Dropout(p=config["lcm"]["model"]["dropout"]))
        modules.append(nn.LeakyReLU())

        for _ in range(self.params["num_layers"]):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.Dropout(p=config["lcm"]["model"]["dropout"]))
            modules.append(nn.LeakyReLU())

        out_dim = len(config["lcm"]["output_features"])
        modules.append(nn.Linear(hidden_dim, out_dim))

        self.cost_layer = nn.Sequential(*modules)
        self.cost_layer.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        out = self.cost_layer(x)

        return out
