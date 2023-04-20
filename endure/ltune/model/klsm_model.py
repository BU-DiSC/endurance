import torch
from torch import nn


class KLSMTuner(nn.Module):
    def __init__(self, config: dict[str, ...]):
        super().__init__()
        self.params = config["learned_tuner"]["models"]["klsm"]

        self.modules = []
        # for _ in range(self.params['hidden_layers']):
        #     modules.append(nn.Linear())

    def forward(self, x) -> torch.Tensor:
        out = x

        return out
