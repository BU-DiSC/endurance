import torch
from torch import nn
from typing import Any


class ClassicTuner(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.params = config["ltune"]["model"]["classic"]
        size_ratio_range = (
            config["lsm"]["size_ratio"]["max"] - config["lsm"]["size_ratio"]["min"] + 1
        )
        self.size_ratio_scale = size_ratio_range - 1
        self.scale_size_ratio = self.params["scale_size_ratio"]
        self.categorical_size_ratio = config["ltune"]["model"]["categorical"]

        in_dim = len(config["ltune"]["input_features"])
        hidden_dim = self.params["layer_size"]
        modules = []

        if self.params["normalize"] == "Layer":
            modules.append(nn.LayerNorm(in_dim))
        if self.params["normalize"] == "Batch":
            modules.append(nn.BatchNorm1d(in_dim))
        # else: No normalization layer

        modules.append(nn.Linear(in_dim, hidden_dim))
        modules.append(nn.Dropout(p=config["ltune"]["model"]["dropout"]))
        modules.append(nn.LeakyReLU())

        for _ in range(self.params["num_layers"]):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.Dropout(p=config["ltune"]["model"]["dropout"]))
            modules.append(nn.LeakyReLU())

        self.sigmoid = nn.Sigmoid()
        self.bits = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        self.bits.apply(self.init_weights)

        if self.categorical_size_ratio:
            size_ratio_out = size_ratio_range
        else:
            size_ratio_out = 1

        self.size_ratio = nn.Sequential(
            nn.Linear(hidden_dim, size_ratio_out),
        )
        self.size_ratio.apply(self.init_weights)
        self.layers = nn.Sequential(*modules)
        self.layers.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x, temp=1e-3, hard=False) -> torch.Tensor:
        out = self.layers(x)
        h = self.bits(out)

        size_ratio = self.size_ratio(out)
        if self.scale_size_ratio:
            size_ratio = self.size_ratio_scale * self.sigmoid(size_ratio)
        if self.categorical_size_ratio:
            size_ratio = nn.functional.gumbel_softmax(size_ratio, tau=temp, hard=hard)
            # size_ratio = size_ratio.softmax(dim=-1)

        return torch.concat([h, size_ratio], dim=-1)
