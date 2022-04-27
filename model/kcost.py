import torch
from torch import nn


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self,  x):
        x = x - self.mean
        x = x / self.std
        return x


class KCostModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.params = config['hyper_params']
        self.params.update(config['static_params'])
        num_feat = self.params['num_cont_vars'] + self.params['embedding_size']
        padding_shape = (0, self.params['embedding_size'])
        mean = torch.nn.functional.pad(
                torch.tensor(self.params['mean_bias']),
                padding_shape,
                mode='constant',
                value=0)
        std = torch.nn.functional.pad(
                torch.tensor(self.params['std_bias']),
                padding_shape,
                mode='constant',
                value=1)
        modules = [Normalize(mean, std)]
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
                self.params['num_cate_vars'] * self.params['max_size_ratio'],
                self.params['embedding_size']
            ),
            nn.ReLU(),
        )
        nn.init.xavier_normal_(self.encode_layer[0].weight)

    def forward(self, x):
        out = self.encode_layer(x[:, self.params['num_cont_vars']:])
        out = torch.cat((x[:, :self.params['num_cont_vars']], out), -1)
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
