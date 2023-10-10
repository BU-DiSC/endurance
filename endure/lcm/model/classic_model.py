from typing import Any

from torch import nn
import torch
import torch.nn.functional as F


class ClassicModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.params = config["lcm"]["model"]["classic"]
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

        self.policy_embedding = nn.Sequential(
            nn.Linear(2, self.params["policy_embedding_size"])
        )
        self.policy_embedding.apply(self.init_weights)

        in_dim = (
            self.num_features
            - 2
            + self.params["policy_embedding_size"]
            + self.params["embedding_size"]
        )
        if self.params["normalize"] == "Layer":
            modules.append(nn.LayerNorm(in_dim))
        elif self.params["normalize"] == "Batch":
            modules.append(nn.BatchNorm1d(in_dim))
        # else: No normalization layer

        hidden_dim = self.params["layer_size"]
        modules.append(nn.Linear(in_dim, hidden_dim))
        modules.append(nn.Dropout(p=config["lcm"]["model"]["dropout"]))
        modules.append(nn.ReLU())

        for _ in range(self.params["num_layers"]):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.Dropout(p=config["lcm"]["model"]["dropout"]))
            modules.append(nn.ReLU())

        out_dim = len(config["lcm"]["output_features"])
        modules.append(nn.Linear(hidden_dim, out_dim))

        self.cost_layer = nn.Sequential(*modules)
        self.cost_layer.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def split_inputs(self, x):
        categorical_bound = self.num_features - 2
        feats = x[:, :categorical_bound]

        if self.training:
            size_ratio = x[:, categorical_bound + 1 :]
            size_ratio = size_ratio.to(torch.long)
            size_ratio = F.one_hot(size_ratio, num_classes=self.size_ratio_range)
            size_ratio = torch.flatten(size_ratio, start_dim=1)
            policy = x[:, categorical_bound : categorical_bound + 1]
            policy = policy.to(torch.long)
            policy = F.one_hot(policy, num_classes=2)
            policy = torch.flatten(policy, start_dim=1)
        else:
            policy = x[:, categorical_bound : categorical_bound + 2]
            size_ratio = x[:, categorical_bound + 2 :]

        return feats, policy, size_ratio

    def forward(self, x):
        feats, policy, size_ratio = self.split_inputs(x)
        size_ratio = size_ratio.to(torch.float)
        size_ratio = self.embedding(size_ratio)

        policy = policy.to(torch.float)
        policy = self.policy_embedding(policy)

        inputs = torch.cat([feats, policy, size_ratio], dim=-1)
        out = self.cost_layer(inputs)

        return out
