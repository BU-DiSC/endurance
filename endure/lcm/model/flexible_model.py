from typing import Callable, Optional, Tuple

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F

OUT_WIDTH = 4

class FlexModel(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        embedding_size: int = 8,
        hidden_length: int = 0,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        width = (num_feats - 2) + (2 * embedding_size)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.t_embedding = nn.Linear(capacity_range, embedding_size)
        self.k_embedding = nn.Linear(capacity_range, embedding_size)
        self.in_norm = norm_layer(width)
        self.in_layer = nn.Linear(width, hidden_width)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_percentage)
        hidden = []
        hidden.append(nn.Identity())
        for _ in range(hidden_length):
            hidden.append(nn.Linear(hidden_width, hidden_width))
            hidden.append(nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(*hidden)
        self.out_layer = nn.Linear(hidden_width, OUT_WIDTH)
        self.capacity_range = capacity_range
        self.num_feats = num_feats

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def _split_input(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        categorical_bound = self.num_feats - 2
        feats = x[:, :categorical_bound]
        capacities = x[:, categorical_bound:]

        if self.training:
            capacities = capacities.to(torch.long)
            capacities = F.one_hot(capacities, num_classes=self.capacity_range)
        else:
            capacities = torch.unflatten(capacities, 1, (-1, self.capacity_range))

        size_ratio = capacities[:, 0, :]
        q_cap = capacities[:, 1, :]

        return (feats, size_ratio, q_cap)

    def _forward_impl(self, x: Tensor) -> Tensor:
        feats, size_ratio, k_cap = self._split_input(x)

        size_ratio = size_ratio.to(torch.float)
        size_ratio = self.t_embedding(size_ratio)

        k_cap = k_cap.to(torch.float)
        k_cap = self.k_embedding(k_cap)

        inputs = torch.cat([feats, size_ratio, k_cap], dim=-1)

        out = self.in_norm(inputs)
        out = self.in_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)
        out = self.out_layer(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out
