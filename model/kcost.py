import torch
from torch import nn


class KCostModel(nn.Module):
    def __init__(self, config, normalize_vals=None):
        super().__init__()
        self.params = config['hyper_params']
        num_feat = self.params['num_cont_vars'] + self.params['num_cate_vars']
        modules = []
        for _ in range(self.params['hidden_layers']):
            modules.append(nn.Linear(num_feat, num_feat))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_feat, 1))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.ReLU())

        self.cost_layer = nn.Sequential(*modules)
        self.encode_layer = nn.Sequential(
            nn.Linear(
                self.params['num_cate_vars'] * config['static_params']['max_size_ratio'],
                self.params['num_cate_vars']
            ),
            nn.ReLU(),
        )
        nn.init.xavier_normal_(self.encode_layer[0].weight)

    def forward(self, x):
        out = self.encode_layer(x[:, self.params['num_cont_vars']:])
        out = torch.cat((out, x[:, :self.params['num_cont_vars']]), -1)
        out = self.cost_layer(out)

        return out


class KCost1Hidden(nn.Module):
    def __init__(self):
        super(KCost1Hidden, self).__init__()
        self.categorical_stack = nn.Sequential(
            nn.Linear(800, 16),
            nn.ReLU(),
            )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.categorical_stack(x[:, 5:])
        out = torch.cat((out, x[:, :5]), -1)
        out = self.linear_relu_stack(out)

        return out
