import os
import toml
import logging
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import Optional, Union
from tqdm import tqdm
from data.parquet_batch_dataset import ParquetBatchDataSet


class Trainer:
    def __init__(
            self,
            config: dict,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            train_data: Union[DataLoader, Dataset, ParquetBatchDataSet],
            test_data: Union[DataLoader, Dataset, ParquetBatchDataSet],
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,):
        self.config = config
        self.log = logging.getLogger(self.config['log']['name'])
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data = test_data
        self.train_len = self.test_len = 0
        self.scheduler = scheduler
        self._early_stop_ticks = 0
        self._move_to_available_device()

    def _move_to_available_device(self):
        use_gpu = (self.config['train']['use_gpu_if_avail'] and
                   torch.cuda.is_available())
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')

        self.log.info(f'Training on device: {self.device}')
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

    def _train_step(self, label, features) -> float:
        label = label.to(self.device)
        features = features.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(features)
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()

        return loss

    def _train_loop(self):
        self.model.train()
        if self.train_len == 0:
            pbar = tqdm(self.train_data, ncols=80)
        else:
            pbar = tqdm(self.train_data, ncols=80, total=self.train_len)

        total_loss = 0
        for batch, (labels, features) in enumerate(pbar):
            loss = self._train_step(labels, features)
            if batch % (100) == 0:
                pbar.set_description(f'loss {loss:e}')
            total_loss += loss
            if self.scheduler is not None:
                self.scheduler.step()
        if type(self.train_data) is ParquetBatchDataSet:
            self.train_data.reset()

        if self.train_len == 0:
            self.train_len = batch + 1

        return total_loss.item() / self.train_len

    def _test_step(self, labels, features):
        with torch.no_grad():
            labels = labels.to(self.device)
            features = features.to(self.device)
            pred = self.model(features)
            test_loss = self.loss_fn(pred, labels).item()

        return test_loss

    def _test_loop(self):
        self.model.eval()
        test_loss = 0
        if self.test_len == 0:
            pbar = tqdm(self.test_data, desc='testing', ncols=80)
        else:
            pbar = tqdm(self.test_data, desc='testing',
                        ncols=80, total=self.test_len)
        for batch, (labels, features) in enumerate(pbar):
            test_loss += self._test_step(labels, features)

        if type(self.test_data) is ParquetBatchDataSet:
            self.test_data.reset()

        if self.test_len == 0:
            self.test_len = batch + 1  # Last batch will correspond to total
        test_loss /= (batch + 1)

        return test_loss

    def _dumpconfig(self, save_dir):
        with open(os.path.join(save_dir, 'config.toml'), 'w') as fid:
            toml.dump(self.config, fid)

    def _checkpoint(self, save_dir, epoch, loss):
        save_pt = {'epoch': epoch,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   'loss': loss}
        torch.save(save_pt, os.path.join(save_dir, f'epoch_{epoch}.checkpoint'))

    def _save_model(self, save_dir, name):
        torch.save(self.model.state_dict(),
                   os.path.join(save_dir, name))

    def _track_early_stop(self, prev_loss: float, curr_loss: float) -> bool:
        """
        Tracking step to check for early stop condition

        :param prev_loss float: loss from prev iteration
        :param curr_loss float: loss at current iteration
        :rtype bool: true if we have met early stop condition, false otherwise
        """
        early_stop_num = self.config['train']['early_stop']['threshold']
        epsilon = self.config['train']['early_stop']['epsilon']

        self.log.info(f'EarlyStop: [{self._early_stop_ticks}/{early_stop_num}]')
        if curr_loss - prev_loss > -epsilon:
            self._early_stop_ticks += 1

        if self._early_stop_ticks >= early_stop_num:
            self.log.info(f'Loss has only changed by {epsilon} for '
                          f'{early_stop_num} epochs. Terminating...')
            return True

        return False

    def run(self):
        max_epochs = self.config['train']['max_epochs']
        save_dir = os.path.join(self.config['io']['data_dir'],
                                self.config['model']['dir'])
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._dumpconfig(save_dir)
        self.log.info('Model parameters')
        for key in self.config['model'].keys():
            self.log.info(f'{key} = {self.config["model"][key]}')
        self.log.info('Training parameters')
        for key in self.config['train'].keys():
            self.log.info(f'{key} = {self.config["train"][key]}')

        df = []
        prev_loss = float('inf')
        loss_min = float('inf')
        for epoch in range(max_epochs):
            self.log.info(f'Epoch: [{epoch+1}/{max_epochs}]')
            train_loss = self._train_loop()
            curr_loss = self._test_loop()
            self.log.info(f'Train loss: {train_loss}')
            self.log.info(f'Test loss: {curr_loss}')
            self._checkpoint(checkpoint_dir, epoch, curr_loss)

            if curr_loss < loss_min:
                loss_min = curr_loss
                self.log.info('New minmum loss, saving...')
                self._save_model(save_dir, 'best.model')
            df.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': curr_loss})
            pd.DataFrame(df).to_csv(
                os.path.join(save_dir, 'losses.csv'),
                index=False)
            if self.config['train']['early_stop']['enabled']:
                self._track_early_stop(prev_loss, curr_loss)
            prev_loss = curr_loss

        self.log.info('Training finished')
