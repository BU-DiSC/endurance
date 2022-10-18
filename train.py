import os
import torch
import logging
import toml
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from model.kcost import KCostModel
import torchdata.datapipes as DataPipe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = toml.load('endure.toml')

logging.basicConfig(
    level=logging.INFO,
    format=cfg['log']['format'],
    datefmt=cfg['log']['datefmt'])
log = logging.getLogger(cfg['log']['name'])
log.info(f'Running with {device=}')

model_dir = os.path.join(cfg['io']['data_dir'], cfg['model']['dir'])
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, 'endure.toml'), 'w') as fid:
    toml.dump(cfg, fid)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    pbar = tqdm(dataloader, ncols=80)
    for batch, data in enumerate(pbar):
        label = data['label'].to(device)
        input = data['feature'].to(device)
        pred = model(input)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % (1000) == 0:
            pbar.set_description(f'loss {loss:>5f}')

    return


def test_loop(dataloader, model, loss_fn):
    test_loss = 0

    model.eval()
    pbar = tqdm(dataloader, desc='validate model', ncols=80)
    with torch.no_grad():
        for batch, data in enumerate(pbar):
            label = data['label'].to(device)
            feature = data['feature'].to(device)
            pred = model(feature)
            test_loss += loss_fn(pred, label).item()

    test_loss /= batch
    log.info(f'validation loss: {test_loss:>8f}')

    return test_loss


def process_row(row: str) -> dict:
    labels = torch.from_numpy(np.array(row[0:4], np.float32))
    features = np.array(row[4:], np.float32)

    # First 4 are h, z0, z1, w, q
    # TODO: Streamline this process
    continuous_data = features[0:5]
    continuous_data -= np.array(cfg['train']['mean_bias'], np.float32)
    continuous_data /= np.array(cfg['train']['std_bias'], np.float32)
    continuous_data = torch.from_numpy(continuous_data)

    # Remaining will be T and Ks
    categorical_data = torch.from_numpy(features[5:])
    features = torch.cat((continuous_data, categorical_data))

    return {'label': labels, 'feature': features}


def file_filter(fname: str) -> bool:
    return fname.endswith('.csv')


train_dir = os.path.join(cfg['io']['data_dir'], cfg['train']['dir'])
test_dir = os.path.join(cfg['io']['data_dir'], cfg['test']['dir'])

dp_train = (DataPipe
            .iter
            .FileLister(train_dir)
            .filter(filter_fn=file_filter)
            .open_files(mode='rt')
            .parse_csv(delimiter=',', skip_lines=1)
            .map(process_row))
if cfg['train']['shuffle'] is True:
    dp_train = dp_train.shuffle()

dp_test = (DataPipe
           .iter
           .FileLister(train_dir)
           .filter(filter_fn=file_filter)
           .open_files(mode='rt')
           .parse_csv(delimiter=',', skip_lines=1)
           .map(process_row))
if cfg['train']['shuffle'] is True:
    dp_train = dp_train.shuffle()

train = DataLoader(
        dp_train,
        batch_size=cfg['train']['batch_size'],
        drop_last=cfg['train']['drop_last'])
test = DataLoader(
        dp_test,
        batch_size=cfg['test']['batch_size'],
        drop_last=cfg['test']['drop_last'])

loss_fn = nn.MSELoss()
model = KCostModel(cfg).to(device)

log.info(f'Model: {cfg["model"]}')
log.info(f'Training params: {cfg["train"]}')

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
    log.info(f'Epoch [{t+1}/{MAX_EPOCHS}] '
             f'Early stop [{no_improvement}/{cfg["train"]["early_stop_num"]}]')
    train_loop(train, model, loss_fn, optimizer)
    scheduler.step()
    curr_loss = test_loop(test, model, loss_fn)
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
