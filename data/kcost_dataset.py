import sys
from pathlib import Path
import torch
import pandas as pd
import logging
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from model.kcost import KCost1Hidden

MAX_LEVELS = 15
KDATA_PATH = '/Users/ndhuynh/sandbox/data/cost_surface_k.feather'

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger()


class KCostDataSet(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        df = pd.read_feather(data_path)

        cont_inputs = ['h', 'z0', 'z1', 'q', 'w']
        cate_inputs = ['T'] + [f'K_{i}' for i in range(MAX_LEVELS)]
        output_cols = ['new_cost']

        mean = df[cont_inputs].mean()
        std = df[cont_inputs].std()
        std[std == 0] = 1
        df[cont_inputs] = (df[cont_inputs] - mean) / std

        cont_inputs = torch.from_numpy(df[cont_inputs].values).float()
        categories = torch.from_numpy(df[cate_inputs].values).to(torch.int64)
        categories = torch.flatten(nn.functional.one_hot(categories, num_classes=50), start_dim=-2)

        self.inputs = torch.cat([cont_inputs, categories], dim=1)
        self.outputs = torch.from_numpy(df[output_cols].values).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        label = self.outputs[idx]

        return inputs, label


def train_loop(dataloader, model, loss_fn, optimizer):
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
data = KCostDataSet(KDATA_PATH)
val_len = int(len(data) * 0.1)
train_len = len(data) - val_len

log.info(f'Splitting dataset train: {train_len}, val: {val_len}')
train, val = torch.utils.data.random_split(data, [train_len, val_len])
train = DataLoader(train, batch_size=32, shuffle=True)
val = DataLoader(val, batch_size=32, shuffle=False)

loss_fn = nn.MSELoss()
model = KCost1Hidden()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# optimizer = torch.optim.Adam(model.parameters())

MAX_EPOCHS = 128
prev_loss = float('inf')
loss_min = float('inf')
no_improvement = 0
for t in range(MAX_EPOCHS):
    log.info(f"Epoch {t+1}/{MAX_EPOCHS} : No Improvement Count {no_improvement}")
    train_loop(train, model, loss_fn, optimizer)
    scheduler.step()
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
