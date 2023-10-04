from typing import Any

from torch import nn
import torch
import torch.nn.functional as F


class QModel(nn.Module):
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

        self.embedding = nn.Sequential(
            nn.Linear(self.size_ratio_range, self.params["embedding_size"])
        )
        self.embedding.apply(self.init_weights)

        self.q_embedding = nn.Sequential(
            nn.Linear(self.size_ratio_range, self.params["embedding_size"])
        )
        self.q_embedding.apply(self.init_weights)

        # Embedding size incorporates T and Ks
        in_dim = self.num_features - 2
        in_dim += 2 * self.params["embedding_size"]

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
        # modules.append(nn.ReLU())

        self.cost_layer = nn.Sequential(*modules)
        self.cost_layer.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def split_inputs(self, x):
        categorical_bound = self.num_features - 2
        feats = x[:, :categorical_bound]
        capacities = x[:, categorical_bound:]

        if self.training:
            capacities = capacities.to(torch.long)
            capacities = F.one_hot(capacities, num_classes=self.size_ratio_range)
        else:
            capacities = torch.unflatten(capacities, 1, (-1, self.size_ratio_range))

        size_ratio = capacities[:, 0, :]
        q_cap = capacities[:, 1, :]

        return feats, size_ratio, q_cap

    def forward(self, x):
        feats, size_ratio, q_cap = self.split_inputs(x)

        size_ratio = size_ratio.to(torch.float)
        size_ratio = self.embedding(size_ratio)

        q_cap = q_cap.to(torch.float)
        q_cap = self.q_embedding(q_cap)

        inputs = torch.cat([feats, size_ratio, q_cap], dim=-1)
        out = self.cost_layer(inputs)

        return out
