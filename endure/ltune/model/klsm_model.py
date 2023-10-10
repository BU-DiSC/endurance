from typing import Any

from torch import nn
import torch


class KLSMTuner(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.params = config["learned_tuner"]["models"]["klsm"]
        # for _ in range(self.params['hidden_layers']):
        #     modules.append(nn.Linear())

    def forward(self, x) -> torch.Tensor:
        out = x

        return out
