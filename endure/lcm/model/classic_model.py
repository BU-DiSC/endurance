from typing import Callable, Optional, Tuple

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F


class ClassicModel(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        embedding_size: int = 8,
        hidden_length: int = 0,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        out_width: int = 4,
        policy_embedding_size: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        width = (num_feats - 2) + embedding_size + policy_embedding_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.t_embedding = nn.Linear(capacity_range, embedding_size)
        self.policy_embedding = nn.Linear(2, policy_embedding_size)
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
        self.out_layer = nn.Linear(hidden_width, out_width)
        self.capacity_range = capacity_range
        self.num_feats = num_feats

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def _split_input(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        t_boundary = self.num_feats - 2
        policy_boundary = t_boundary + self.capacity_range
        feats = x[:, :t_boundary]

        if self.training:
            size_ratio = x[:, t_boundary : policy_boundary]
            size_ratio = size_ratio.to(torch.long)
            size_ratio = F.one_hot(size_ratio, num_classes=self.capacity_range)
            size_ratio = torch.flatten(size_ratio, start_dim=1)
            policy = x[:, policy_boundary : policy_boundary + 1]
            policy = policy.to(torch.long)
            policy = F.one_hot(policy, num_classes=2)
            policy = torch.flatten(policy, start_dim=1)
        else:
            policy = x[:, policy_boundary : policy_boundary + 2]
            size_ratio = x[:, t_boundary : policy_boundary]

        return (feats, size_ratio, policy)

    def _forward_impl(self, x: Tensor) -> Tensor:
        feats, size_ratio, policy = self._split_input(x)
        size_ratio = size_ratio.to(torch.float)
        size_ratio = self.t_embedding(size_ratio)

        policy = policy.to(torch.float)
        policy = self.policy_embedding(policy)

        inputs = torch.cat([feats, size_ratio, policy], dim=-1)

        out = self.in_layer(inputs)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)
        out = self.out_layer(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_impl(x)

        return out
