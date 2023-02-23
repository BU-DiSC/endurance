import os
import toml
import logging
import torch
import csv
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union
from tqdm import tqdm
from endure.lcm.data.parquet_batch_dataset import ParquetBatchDataSet


class Trainer:
    def __init__(
            self,
            config: dict[str, ...],
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            train_data: Union[DataLoader, Dataset, ParquetBatchDataSet],
            test_data: Union[DataLoader, Dataset, ParquetBatchDataSet],
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        self._config = config
        self.log = logging.getLogger(self._config['log']['name'])
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data = test_data
        self.train_len = self.test_len = 0
        self.scheduler = scheduler
        self._early_stop_ticks = 0
        self._move_to_available_device()

    def _move_to_available_device(self) -> None:
        use_gpu = (self._config['train']['use_gpu_if_avail'] and
                   torch.cuda.is_available())
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')

        self.log.info(f'Training on device: {self.device}')
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        return

    def _train_step(self, label, features) -> float:
        label = label.to(self.device)
        features = features.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(features)
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()

        return loss

    def _train_loop(self) -> float:
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

    def _test_step(
            self,
            labels: torch.Tensor,
            features: torch.Tensor) -> float:
        with torch.no_grad():
            labels = labels.to(self.device)
            features = features.to(self.device)
            pred = self.model(features)
            test_loss = self.loss_fn(pred, labels).item()

        return test_loss

    def _test_loop(self) -> float:
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

    def _dumpconfig(self, save_dir: str) -> None:
        with open(os.path.join(save_dir, 'endure.toml'), 'w') as fid:
            toml.dump(self._config, fid)

        return

    def _checkpoint(self, save_dir: str, epoch: int, loss) -> None:
        save_pt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss}
        torch.save(
            save_pt,
            os.path.join(save_dir, f'epoch_{epoch:02f}.checkpoint'))

        return

    def _save_model(self, save_dir: str, name: str) -> None:
        torch.save(self.model.state_dict(),
                   os.path.join(save_dir, name))

        return

    def run(self) -> None:
        max_epochs = self._config['train']['max_epochs']
        save_dir = os.path.join(self._config['io']['data_dir'],
                                self._config['train']['save_dir'])
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._dumpconfig(save_dir)
        self.log.info('Training parameters')
        for key in self._config['train'].keys():
            self.log.info(f"{key} = {self._config['train'][key]}")

        with open(os.path.join(save_dir, 'losses.csv'), 'w') as fid:
            loss_csv_write = csv.writer(fid)
            loss_csv_write.writerow(['epoch', 'train_loss', 'test_loss'])
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
            with open(os.path.join(save_dir, 'losses.csv'), 'a') as fid:
                write = csv.writer(fid)
                write.writerow([epoch, train_loss, curr_loss])

        self.log.info('Training finished')

        return
