from typing import Callable, Optional
import sys

import numpy as np
from torch import Tensor, nn
import torch
from reinmax import reinmax

class KapDecision(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_kap: int,
        categorical_mode: str = 'gumbel',
    ) -> None:
        super().__init__()
        self.decision_layers = nn.ModuleList(
            [nn.Linear(input_size, num_classes) for _ in range(num_kap)]
        )
        self.categorical_mode = categorical_mode

    def _forward_impl(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = []
        for layer in self.decision_layers:
            k = layer(x)
            if self.categorical_mode == 'reinmax':
                k, _ = reinmax(k, tau=temp)
            else: # categorical_mode == 'gumbel'
                k = nn.functional.gumbel_softmax(k, tau=temp, hard=hard)
            out.append(k)
        out = torch.stack(out, dim=1)

        return out

    def forward(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self._forward_impl(x, temp=temp, hard=hard)

        return out

class KapLSMTuner(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        hidden_length: int = 1,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        num_kap: int = 10,
        categorical_mode: str = 'gumbel',
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.in_norm = norm_layer(num_feats)
        self.in_layer = nn.Linear(num_feats, hidden_width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percentage)
        hidden = []
        for _ in range(hidden_length):
            hidden.append(nn.Linear(hidden_width, hidden_width))
        self.hidden = nn.Sequential(*hidden)

        self.k_path = nn.Linear(hidden_width, hidden_width)
        self.t_path = nn.Linear(hidden_width, hidden_width)
        self.bits_path = nn.Linear(hidden_width, hidden_width)

        self.k_decision = KapDecision(hidden_width, capacity_range, num_kap)
        self.t_decision = nn.Linear(hidden_width, capacity_range)
        self.bits_decision = nn.Linear(hidden_width, 1)

        self.capacity_range = capacity_range
        self.num_feats = num_feats
        self.num_kap = num_kap
        self.categorical_mode = categorical_mode

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def calc_max_level(
        self,
        x: Tensor,  # input tensor
        bpe: Tensor,
        size_ratio: Tensor,
    ) -> Tensor:
        # KLSM: ["z0", "z1", "q", "w", "B", "s", "E", "H", "N"]
        # IDX:  [  0 ,   1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8]
        size_ratio = torch.squeeze(torch.argmax(size_ratio, dim=-1))
        bpe = torch.squeeze(torch.clone(bpe))
        bpe[bpe < 0] = 0
        max_bits = x[:, 7]    # H
        num_elem = x[:, 8]    # N
        entry_size = x[:, 6]  # E
        bpe[bpe > max_bits] = max_bits[bpe > max_bits] - 0.1
        mbuff = (max_bits - bpe) * num_elem
        level = torch.log(((num_elem * entry_size) / mbuff) + 1) / torch.log(size_ratio)
        level = torch.ceil(level)

        return level

    def _forward_impl(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self.in_norm(x)
        out = self.in_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)

        bits_out = self.bits_path(out)
        bits = self.bits_decision(bits_out)

        t_out = self.t_path(out)
        t = self.t_decision(t_out)
        if self.categorical_mode == 'reinmax':
            t, _ = reinmax(t, tau=temp)
        else: # categorical_mode == 'gumbel'
            t = nn.functional.gumbel_softmax(t, tau=temp, hard=hard)

        k_out = self.k_path(out)
        k = self.k_decision(k_out, temp=temp, hard=hard)
        k = torch.flatten(k, start_dim=1)

        # max_levels = self.calc_max_level(x, bits, t)
        # max_levels = max_levels.to(torch.long)
        # level_blanks = 1 - nn.functional.one_hot(max_levels).cumsum(dim=1)

        out = torch.concat([bits, t, k], dim=-1)

        return out

    def forward(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self._forward_impl(x, temp=temp, hard=hard)

        return out
