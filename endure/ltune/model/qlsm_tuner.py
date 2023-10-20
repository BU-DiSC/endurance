from typing import Callable, Optional

from torch import Tensor, nn
import torch


class QLSMTuner(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        hidden_length: int = 1,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.in_norm = norm_layer(num_feats)
        self.in_layer = nn.Linear(num_feats, hidden_width)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_percentage)
        hidden = []
        for _ in range(hidden_length):
            hidden.append(nn.Linear(hidden_width, hidden_width))
        self.hidden = nn.Sequential(*hidden)

        self.q_decision = nn.Linear(hidden_width, capacity_range)
        self.t_decision = nn.Linear(hidden_width, capacity_range)
        self.bits_decision = nn.Linear(hidden_width, 1)

        self.capacity_range = capacity_range
        self.num_feats = num_feats

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def _forward_impl(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self.in_norm(x)
        out = self.in_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)
        out = self.relu(out)

        bits = self.bits_decision(out)
        q = self.q_decision(out)
        q = nn.functional.gumbel_softmax(q, tau=temp, hard=hard)
        t = self.t_decision(out)
        t = nn.functional.gumbel_softmax(t, tau=temp, hard=hard)

        out = torch.concat([bits, t, q], dim=-1)

        return out

    def forward(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self._forward_impl(x, temp=temp, hard=hard)

        return out

# class QLSMTuner(nn.Module):
#     def __init__(self, config: dict[str, Any]):
#         super().__init__()
#         self.params = config["ltune"]["model"]["classic"]
#         size_ratio_config = config["lsm"]["size_ratio"]
#         size_ratio_range = size_ratio_config["max"] - size_ratio_config["min"] + 1
#
#         in_dim = len(config["ltune"]["input_features"])
#         hidden_dim = self.params["layer_size"]
#         modules = []
#
#         if self.params["normalize"] == "Layer":
#             modules.append(nn.LayerNorm(in_dim))
#         if self.params["normalize"] == "Batch":
#             modules.append(nn.BatchNorm1d(in_dim))
#         # else: No normalization layer
#
#         modules.append(nn.Linear(in_dim, hidden_dim))
#         modules.append(nn.Dropout(p=config["ltune"]["model"]["dropout"]))
#         modules.append(nn.LeakyReLU())
#
#         for _ in range(self.params["num_layers"]):
#             modules.append(nn.Linear(hidden_dim, hidden_dim))
#             modules.append(nn.Dropout(p=config["ltune"]["model"]["dropout"]))
#             modules.append(nn.LeakyReLU())
#
#         self.layers = nn.Sequential(*modules)
#         self.layers.apply(self.init_weights)
#
#         self.q = nn.Linear(hidden_dim, size_ratio_range)
#         nn.init.xavier_normal_(self.q.weight)
#
#         self.bits = nn.Linear(hidden_dim, 1)
#         nn.init.xavier_normal_(self.bits.weight)
#
#         self.size_ratio = nn.Linear(hidden_dim, size_ratio_range)
#         nn.init.xavier_normal_(self.size_ratio.weight)
#
#     def init_weights(self, layer):
#         if isinstance(layer, nn.Linear):
#             nn.init.xavier_normal_(layer.weight)
#
#     def forward(self, x, temp=1e-3, hard=False) -> torch.Tensor:
#         out = self.layers(x)
#         h = self.bits(out)
#         q = self.q(out)
#         q = nn.functional.gumbel_softmax(q, tau=temp, hard=hard)
#         size_ratio = self.size_ratio(out)
#         size_ratio = nn.functional.gumbel_softmax(size_ratio, tau=temp, hard=hard)
#
#         out = torch.concat([h, size_ratio, q], dim=-1)
#
#         return out
