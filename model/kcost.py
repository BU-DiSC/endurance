from torch import nn


class KCost1Hidden(nn.Module):
    def __init__(self):
        super(KCost1Hidden, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


class KCost2Hidden(nn.Module):
    def __init__(self):
        super(KCost2Hidden, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
