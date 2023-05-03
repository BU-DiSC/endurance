from typing import Any

from torch import nn
import torch


class ClassicModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.params = config["lcm"]["model"]["classic"]
        in_dim = self.params["num_cont_vars"] + self.params["embedding_size"]
        hidden_dim = self.params["layer_size"]
        out_dim = len(config["lcm"]["output_features"])
        modules = []

        self.embedding = nn.Embedding(
            num_embeddings=(
                config["lsm"]["size_ratio"]["max"]
                - config["lsm"]["size_ratio"]["min"]
                + 1
            ),
            embedding_dim=self.params["embedding_size"],
            max_norm=True,
        )

        if self.params["normalize"] == "Layer":
            modules.append(nn.LayerNorm(in_dim))
        elif self.params["normalize"] == "Batch":
            modules.append(nn.BatchNorm1d(in_dim))
        # else: No normalization layer

        modules.append(nn.Linear(in_dim, hidden_dim))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.Dropout(p=config["lcm"]["model"]["dropout"]))
        modules.append(nn.LeakyReLU())

        for _ in range(self.params["num_layers"]):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.Dropout(p=config["lcm"]["model"]["dropout"]))
            modules.append(nn.LeakyReLU())

        modules.append(nn.Linear(hidden_dim, out_dim))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.LeakyReLU())

        self.cost_layer = nn.Sequential(*modules)

    def forward(self, x):
        size_ratio = x[:, -1].to(torch.int32)
        out = torch.flatten(self.embedding(size_ratio), start_dim=1)
        out = torch.cat((x[:, : self.params["num_cont_vars"]], out), -1)
        out = self.cost_layer(out)

        return out
