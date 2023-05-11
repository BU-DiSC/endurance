import torch
from torch import nn
from typing import Any


class ClassicTuner(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.params = config["ltune"]["model"]["classic"]
        size_ratio_config = config["lsm"]["size_ratio"]
        size_ratio_range = size_ratio_config["max"] - size_ratio_config["min"] + 1

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

        self.bits = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.bits.weight)

        self.size_ratio = nn.Linear(hidden_dim, size_ratio_range)
        nn.init.xavier_normal_(self.size_ratio.weight)

        self.layers = nn.Sequential(*modules)
        self.layers.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x, temp=1e-3, hard=False) -> torch.Tensor:
        out = self.layers(x)
        h = self.bits(out)

        size_ratio = self.size_ratio(out)
        size_ratio = nn.functional.gumbel_softmax(size_ratio, tau=temp, hard=hard)

        return torch.concat([h, size_ratio], dim=-1)
