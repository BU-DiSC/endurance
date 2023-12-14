from typing import Callable, Optional

from torch import Tensor, nn
import torch

class KapDecision(nn.Module):
    def __init__(self, input_size: int, num_classes: int, num_kap: int) -> None:
        super().__init__()
        decision_layers = []
        for _ in range(num_kap):
            decision_layers.append(nn.Linear(input_size, num_classes))

        self.decision_layers = decision_layers

    def _forward_impl(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = []
        for layer in self.decision_layers:
            k = layer(x)
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

        self.k_decision = KapDecision(hidden_width, capacity_range, num_kap)
        self.t_decision = nn.Linear(hidden_width, capacity_range)
        self.bits_decision = nn.Linear(hidden_width, 1)

        self.capacity_range = capacity_range
        self.num_feats = num_feats
        self.num_kap = num_kap

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def _forward_impl(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self.in_norm(x)
        out = self.in_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)

        bits = self.bits_decision(out)
        k = self.k_decision(out, temp=temp, hard=hard)
        k = torch.flatten(k, start_dim=1)
        t = self.t_decision(out)
        t = nn.functional.gumbel_softmax(t, tau=temp, hard=hard)

        out = torch.concat([bits, t, k], dim=-1)

        return out

    def forward(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self._forward_impl(x, temp=temp, hard=hard)

        return out
