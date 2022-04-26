import os
import torch
import logging
import toml
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from model.kcost import KCostModel
from data.kcost_dataset import KCostDataSetSplit


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger()
config = toml.load('config/training.toml')


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
paths = [os.path.join(config['io']['data_dir'], config['io']['train_dir'], data) for data in config['io']['train_data']]
data = KCostDataSetSplit(config, paths)
val_len = int(len(data) * config['hyper_params']['validate_frac'])
train_len = len(data) - val_len

log.info(f'Splitting dataset train: {train_len}, val: {val_len}')
train, val = torch.utils.data.random_split(data, [train_len, val_len])
train = DataLoader(train, batch_size=config['hyper_params']['batch_size'], shuffle=True)
val = DataLoader(val, batch_size=config['hyper_params']['batch_size'], shuffle=False)

loss_fn = nn.MSELoss()
model = KCostModel(config, data.normalize_vars)
optimizer = torch.optim.SGD(model.parameters(), lr=config['hyper_params']['learning_rate'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['hyper_params']['lr_schedule_gamma'])

MAX_EPOCHS = config['hyper_params']['max_epochs']
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
