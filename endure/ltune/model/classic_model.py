import torch
from torch import nn


class ClassicTuner(nn.Module):
    def __init__(self, config: dict[str, ...]):
        super().__init__()
        self.params = config['ltune']['model']['classic']
        size_ratio_range = (config['lsm']['size_ratio']['max']
                            - config['lsm']['size_ratio']['min']
                            + 1)

        modules = []
        num_feat = config['ltune']['model']['in_dim']
        out_dim = 1 + size_ratio_range
        for _ in range(self.params['hidden_layers']):
            modules.append(nn.Linear(num_feat, num_feat))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_feat, out_dim))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.ReLU())

        self.layers = nn.Sequential(*modules)

    def forward(self, x, temp=0.1, hard=False) -> torch.Tensor:
        out = self.layers(x)
        size_ratio = nn.functional.gumbel_softmax(
            out[:, 1:], tau=temp, hard=hard)

        return torch.concat([out[:, 0].view(-1, 1), size_ratio], dim=-1)
