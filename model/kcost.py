import torch
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Normalize(nn.Module):
    def __init__(self, mean, std, padding_shape=None):
        super().__init__()
        if padding_shape is not None:
            mean = torch.nn.functional.pad(
                mean, padding_shape, mode="constant", value=0)
            std = torch.nn.functional.pad(
                std, padding_shape, mode="constant", value=1)
        self.mean = mean.to(DEVICE)
        self.std = std.to(DEVICE)

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x


class KCostModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.params = config["hyper_params"] | config["static_params"]

        num_feat = self.params["num_cont_vars"] + self.params["embedding_size"]
        padding_shape = (0, self.params["embedding_size"])
        mean = torch.tensor(self.params["mean_bias"])
        std = torch.tensor(self.params["std_bias"])

        self.encode_layer = nn.Sequential(
            nn.Linear(
                self.params["num_cate_vars"] * self.params["max_size_ratio"],
                self.params["embedding_size"],
            ),
            nn.ReLU(),
        )
        nn.init.xavier_normal_(self.encode_layer[0].weight)

        modules = [Normalize(mean, std, padding_shape=padding_shape)]
        for _ in range(self.params["hidden_layers"]):
            modules.append(nn.Linear(num_feat, num_feat))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_feat, 1))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.ReLU())

        self.cost_layer = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encode_layer(x[:, self.params["num_cont_vars"]:])
        out = torch.cat((x[:, : self.params["num_cont_vars"]], out), -1)
        out = self.cost_layer(out)

        return out


class KCostModelAlpha(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.params = config["hyper_params"] | config["static_params"]

        num_feat = self.params["num_cont_vars"] + self.params["embedding_size"]
        padding_shape = (0, self.params["embedding_size"])
        mean = torch.tensor(self.params["mean_bias"])
        std = torch.tensor(self.params["std_bias"])

        self.encode_layer = nn.Sequential(
            nn.Linear(
                self.params["num_cate_vars"] * self.params["max_size_ratio"],
                self.params["embedding_size"],
            ),
            nn.ReLU(),
        )
        nn.init.xavier_normal_(self.encode_layer[0].weight)

        modules = [Normalize(mean, std, padding_shape=padding_shape)]
        for _ in range(self.params["hidden_layers"]):
            modules.append(nn.Linear(num_feat, num_feat))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_feat, self.params["out_dims"]))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(nn.ReLU())

        self.cost_layer = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encode_layer(x[:, self.params["num_cont_vars"]:])
        out = torch.cat((x[:, : self.params["num_cont_vars"]], out), -1)
        out = self.cost_layer(out)

        return out
