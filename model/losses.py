import torch


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return self.mse(torch.log(pred + 1), torch.log(label + 1))


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(label + 1)))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return torch.sqrt(self.mse(pred, label))


class NMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return self.mse(pred, label) / torch.square(torch.sum(label))


class NMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.msle = MSLELoss()

    def forward(self, pred, label):
        return (self.msle(pred, label)
                / torch.sum(torch.square(torch.log(pred + 1))))
