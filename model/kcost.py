import torch
from torch import nn


class KCostModel(nn.Module):
    def __init__(self, num_cont_vars=5, num_categorical_hidden_vars=16, hidden_layers=2):
        super().__init__()
        num_feat = num_cont_vars + num_categorical_hidden_vars
        self.num_cont_vars = num_cont_vars
        modules = []
        for _ in hidden_layers:
            modules.append(nn.Linear(num_feat, num_feat))
            modules.append(nn.ReLU)
        modules.append(nn.Linear(num_feat, 1))
        modules.append(nn.ReLU())
        self.cost_layer = nn.Sequential(*modules)

        self.encode_layer = nn.Sequential(
            nn.Linear(800, num_categorical_hidden_vars),
            nn.ReLU(),
        )

        nn.init.xavier_normal_(self.cost_layer.weight)
        nn.init.xavier_normal_(self.encode_layer.weight)

    def forward(self, x):
        out = self.encode_layer(x[:, self.num_cont_vars:])
        out = torch.cat((out, x[:, :self.num_cont_vars]), -1)
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
