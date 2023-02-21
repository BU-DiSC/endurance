import torch
from torch import nn


class ClassicModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.params = config['lcm']['model']['classic']

        self.embedding = nn.Embedding(
                num_embeddings=config['lsm']['size_ratio']['max'],
                embedding_dim=self.params['embedding_size'],
                max_norm=True)
        num_feat = self.params['num_cont_vars'] + self.params['embedding_size']

        modules = []
        for _ in range(self.params['hidden_layers']):
            modules.append(nn.Linear(num_feat, num_feat))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_feat, self.params['out_dims']))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.ReLU())

        self.cost_layer = nn.Sequential(*modules)

    def forward(self, x):
        size_ratio = x[:, -1].to(torch.int32)
        out = torch.flatten(self.embedding(size_ratio), start_dim=1)
        out = torch.cat((x[:, :self.params['num_cont_vars']], out), -1)
        out = self.cost_layer(out)

        return out
