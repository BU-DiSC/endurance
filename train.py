import os
import torch
import logging
import toml
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from model.kcost import KCostModelAlpha
from data.kcost_dataset import KCostDataSet


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger()
cfg = toml.load('config/training.toml')
model_dir = cfg['io']['model_dir']
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, 'config.toml'), "w") as fid:
    toml.dump(cfg, fid)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    pbar = tqdm(dataloader, ncols=80)
    for batch, (X, y) in enumerate(pbar):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % (10000) == 0:
            pbar.set_description(f'loss {loss:>5f}')

    return


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='validate model', ncols=80):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    log.info(f'validation loss: {test_loss:>8f}')

    return test_loss


log.info('Reading data')
data_dir = os.path.join(cfg['io']['data_dir'], cfg['io']['train_dir'])
paths = []
for data_file in cfg['io']['train_data']:
    paths.append(os.path.join(data_dir, data_file))
data = KCostDataSet(cfg, paths)
val_len = int(len(data) * cfg['validate']['percent'])
train_len = len(data) - val_len

log.info(f'Splitting dataset train: {train_len}, val: {val_len}')
train, val = torch.utils.data.random_split(data, [train_len, val_len])
train = DataLoader(
        train,
        batch_size=cfg['train']['batch_size'],
        shuffle=cfg["train"]["shuffle"])
val = DataLoader(
        val,
        batch_size=cfg['validate']['batch_size'],
        shuffle=cfg["validate"]["shuffle"])

loss_fn = nn.MSELoss()
model = KCostModelAlpha(cfg)
log.info(f"Model params: {cfg['hyper_params']}")
log.info(f"Training params: {cfg['train']}")
optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg['train']['learning_rate'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg['train']['learning_rate_decay'])

MAX_EPOCHS = cfg['train']['max_epochs']
prev_loss = loss_min = float('inf')
no_improvement = 0
for t in range(MAX_EPOCHS):
    log.info(f'Epoch [{t+1}/{MAX_EPOCHS}]'
             f'Early stop [{no_improvement}/{cfg["train"]["early_stop_num"]}]')
    train_loop(train, model, loss_fn, optimizer)
    scheduler.step()
    curr_loss = test_loop(val, model, loss_fn)
    if curr_loss < loss_min:
        loss_min = curr_loss
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, 'kcost_min.model'))
        no_improvement = 0
    else:
        no_improvement += 1

    save_pt = {
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': curr_loss}
    torch.save(save_pt, os.path.join(model_dir, 'checkpoint.pt'))
    if no_improvement > cfg["train"]['early_stop_num']:
        log.info('Early termination, exiting...')
        break
    prev_loss = curr_loss
