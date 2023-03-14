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
        modules.append(nn.Linear(num_feat, out_dim))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.ReLU())
        for _ in range(self.params['hidden_layers']):
            modules.append(nn.Linear(out_dim, out_dim))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.ReLU())

        self.bits = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Hardtanh(
                min_val=config['lsm']['bits_per_elem']['min'],
                max_val=config['lsm']['bits_per_elem']['max'],
            )
        )

        self.size_ratio = nn.Sequential(
            nn.Linear(out_dim, size_ratio_range),
        )
        self.layers = nn.Sequential(*modules)

    def forward(self, x, temp=0.1, hard=False) -> torch.Tensor:
        out = self.layers(x)
        size_ratio = self.size_ratio(out)
        size_ratio = nn.functional.gumbel_softmax(
            size_ratio,
            tau=temp,
            hard=hard
        )
        h = self.bits(out)

        return torch.concat([h, size_ratio], dim=-1)
