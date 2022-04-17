import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader


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


class KCostDouble2Hidden(nn.Module):
    def __init__(self):
        super(KCostDouble2Hidden, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 42),
            nn.ReLU(),
            nn.Linear(42, 42),
            nn.ReLU(),
            nn.Linear(42, 42),
            nn.ReLU(),
            nn.Linear(42, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger()
KDATA_PATH = '/scratchNVM0/ndhuynh/data/cost_surface_k.csv'


class KCostDataSet(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.df = pd.read_csv(KDATA_PATH)
        data = self.df['K'].map(lambda x: list(map(int, x[1:-1].split())))
        Ks = pd.DataFrame(data.to_list()).add_prefix('K_').fillna(0)
        self.df = pd.concat([self.df, Ks], axis=1)

        max_levels = self.df.query('T == 2')['K'].apply(lambda x: len(x[1:-1].split())).max()
        input_cols = ['h', 'T', 'z0', 'z1', 'q', 'w'] + [f'K_{i}' for i in range(max_levels)]
        output_cols = ['new_cost']

        mean = self.df[input_cols].mean()
        std = self.df[input_cols].std()
        std[std == 0] = 1
        self.df[input_cols] = (self.df[input_cols] - mean) / std

        self.inputs = torch.from_numpy(self.df[input_cols].values).float()
        self.outputs = torch.from_numpy(self.df[output_cols].values).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        label = self.outputs[idx]

        return inputs, label


def train_loop(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    model.train()
    pbar = tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
    for batch, (X, y) in enumerate(pbar):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % (10000) == 0:
            pbar.set_description(f'loss: {loss:>7f}')

    return


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    log.info(f'validation loss: {test_loss:>8f}')

    return test_loss


log.info('Reading data')
data = KCostDataSet()
val_len = int(len(data) * 0.1)
train_len = len(data) - val_len

log.info(f'Splitting dataset train: {train_len}, val: {val_len}')
train, val = torch.utils.data.random_split(data, [train_len, val_len])
train = DataLoader(train, batch_size=32, shuffle=True)
val = DataLoader(val, batch_size=32, shuffle=False)

loss_fn = nn.MSELoss()
model = KCost2Hidden()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters())

MAX_EPOCHS = 64
prev_loss = float('inf')
loss_min = float('inf')
no_improvement = 0
for t in range(MAX_EPOCHS):
    log.info(f"Epoch {t+1}/{MAX_EPOCHS} : No Improvement Count {no_improvement}")
    train_loop(train, model, loss_fn, optimizer)
    curr_loss = test_loop(val, model, loss_fn)
    if curr_loss < loss_min:
        loss_min = curr_loss
        torch.save(model.state_dict(), 'kcost_min.model')
        no_improvement = 0
    else:
        no_improvement += 1

    log.info('Check pointing...')
    torch.save({
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': curr_loss,
        },
        'checkpoint.pt')
    if no_improvement > 3:
        log.info('Early termination: No improvement over 4 epochs, early terminating')
        break
    prev_loss = curr_loss
