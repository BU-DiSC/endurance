from typing import Any

from torch import nn
import torch


class FlexibleModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.params = config["lcm"]["model"]["flexible"]
        num_classes = (
            config["lsm"]["size_ratio"]["max"] - config["lsm"]["size_ratio"]["min"] + 1
        )

        self.embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=self.params["embedding_size"],
            max_norm=True,
        )
        embedding_output = self.params["num_cate_vars"] * self.params["embedding_size"]
        num_feat = self.params["num_cont_vars"] + embedding_output

        modules = []
        for _ in range(self.params["hidden_layers"]):
            modules.append(nn.Linear(num_feat, num_feat))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_feat, self.params["out_dims"]))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.ReLU())

        self.cost_layer = nn.Sequential(*modules)

    def forward(self, x) -> torch.Tensor:
        cate_inputs = x[:, self.params["num_cont_vars"] :]
        out = self.embedding(cate_inputs.to(torch.int32))
        out = torch.flatten(out, start_dim=1)
        out = torch.cat([x[:, : self.params["num_cont_vars"]], out], -1)
        out = self.cost_layer(out)

        return out
