from typing import Callable, Optional, Tuple

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F

DECISION_DIM = 64

class KapModel(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        embedding_size: int = 8,
        hidden_length: int = 1,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        max_levels: int = 20,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        width = (num_feats - (max_levels + 1)) + ((max_levels + 1) * embedding_size)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.t_embedding = nn.Linear(capacity_range, embedding_size)
        self.k_embedding = nn.Linear(capacity_range, embedding_size)

        self.in_norm = norm_layer(width)
        self.in_layer = nn.Linear(width, hidden_width)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_percentage)
        hidden = []
        for _ in range(hidden_length):
            hidden.append(nn.Linear(hidden_width, hidden_width))
            hidden.append(nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(*hidden)
        self.out_layer = nn.Linear(hidden_width, DECISION_DIM)
        self.z0 = nn.Linear(int(DECISION_DIM / 4), 1)
        self.z1 = nn.Linear(int(DECISION_DIM / 4), 1)
        self.q = nn.Linear(int(DECISION_DIM / 4), 1)
        self.w = nn.Linear(int(DECISION_DIM / 4), 1)

        self.capacity_range = capacity_range
        self.num_feats = num_feats
        self.max_levels = max_levels

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def _split_input(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        categorical_bound = self.num_feats - (self.max_levels + 1)
        feats = x[:, :categorical_bound]
        capacities = x[:, categorical_bound:]

        if self.training:
            capacities = capacities.to(torch.long)
            capacities = F.one_hot(capacities, num_classes=self.capacity_range)
        else:
            capacities = torch.unflatten(capacities, 1, (-1, self.capacity_range))

        size_ratio = capacities[:, 0, :]
        k_cap = capacities[:, 1:, :]

        return (feats, size_ratio, k_cap)

    def _forward_impl(self, x: Tensor) -> Tensor:
        feats, size_ratio, k_cap = self._split_input(x)

        size_ratio = size_ratio.to(torch.float)
        size_ratio = self.t_embedding(size_ratio)

        k_cap = k_cap.to(torch.float)
        k_cap = self.k_embedding(k_cap)
        k_cap = torch.flatten(k_cap, start_dim=1)

        inputs = torch.cat([feats, size_ratio, k_cap], dim=-1)

        out = self.in_norm(inputs)
        out = self.in_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)
        out = self.out_layer(out)
        head_dim = int(DECISION_DIM / 4)
        z0 = self.z0(out[:, 0:head_dim])
        z1 = self.z1(out[:, head_dim:2*head_dim])
        q = self.q(out[:, 2*head_dim:3*head_dim])
        w = self.w(out[:, 3*head_dim:4*head_dim])
        out = torch.cat([z0, z1, q, w], dim=-1)

        return out

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out
