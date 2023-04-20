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

        in_dim = config["ltune"]["model"]["in_dim"]
        hidden_dim = self.params["layer_size"]
        modules = []

        # Normalize layer
        if self.params["normalize_layer"]:
            modules.append(nn.LayerNorm(in_dim))

        # First layer in
        modules.append(nn.Linear(in_dim, hidden_dim))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.Dropout(p=config["ltune"]["model"]["dropout"]))
        modules.append(nn.ReLU())

        # Hidden layers
        for _ in range(self.params["num_layers"]):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.Dropout(p=config["ltune"]["model"]["dropout"]))
            modules.append(nn.ReLU())

        # Decision layers
        self.bits = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Hardtanh(
                min_val=config["lsm"]["bits_per_elem"]["min"],
                max_val=config["lsm"]["bits_per_elem"]["max"],
            ),
        )
        self.size_ratio = nn.Sequential(
            nn.Linear(hidden_dim, size_ratio_range),
        )

        self.layers = nn.Sequential(*modules)

    def forward(self, x, temp=0.1, hard=False) -> torch.Tensor:
        out = self.layers(x)
        size_ratio = self.size_ratio(out)
        size_ratio = nn.functional.gumbel_softmax(size_ratio, tau=temp, hard=hard)
        h = self.bits(out)

        return torch.concat([h, size_ratio], dim=-1)
