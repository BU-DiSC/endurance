import os
import toml
import logging
import torch
import csv
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        log: logging.Logger,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_data: Union[DataLoader, Dataset],
        test_data: Union[DataLoader, Dataset],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_epochs: Optional[int] = 10,
        base_dir: Optional[str] = './',
        use_gpu_if_avail: Optional[bool] = False,
    ) -> None:
        self.log = log
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data = test_data
        self.train_len = self.test_len = 0
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.base_dir = base_dir
        self.use_gpu_if_avail = use_gpu_if_avail
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir)

        self._early_stop_ticks = 0
        self._move_to_available_device()

    def _move_to_available_device(self) -> None:
        self.device = torch.device('cpu')
        if self.use_gpu_if_avail:
            self.device = torch.device('cuda')

        self.log.info(f'Training on device: {self.device}')
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        return

    def _train_step(
        self,
        label: torch.Tensor,
        features: torch.Tensor,
    ) -> float:
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

        if self.train_len == 0:
            self.train_len = batch + 1

        return total_loss.item() / self.train_len

    def _test_step(
        self,
        labels: torch.Tensor,
        features: torch.Tensor
    ) -> float:
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

        if self.test_len == 0:
            self.test_len = batch + 1  # Last batch will correspond to total
        test_loss /= (batch + 1)

        return test_loss

    def _checkpoint(self, epoch: int, loss) -> None:
        save_pt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        checkpoint_name = os.path.join(
            self.checkpoint_dir,
            f'epoch_{epoch:02f}.checkpoint'
        )
        torch.save(save_pt, checkpoint_name)

        return

    def _save_model(self, name: str) -> None:
        save_path = os.path.join(self.base_dir, name)
        torch.save(self.model.state_dict(), save_path)

        return

    def run(self) -> None:
        with open(os.path.join(self.base_dir, 'losses.csv'), 'w') as fid:
            loss_csv_write = csv.writer(fid)
            loss_csv_write.writerow(['epoch', 'train_loss', 'test_loss'])

        loss_min = float('inf')
        for epoch in range(self.max_epochs):
            self.log.info(f'Epoch: [{epoch+1}/{self.max_epochs}]')
            train_loss = self._train_loop()
            curr_loss = self._test_loop()
            self.log.info(f'Train loss: {train_loss}')
            self.log.info(f'Test loss: {curr_loss}')
            self._checkpoint(epoch, curr_loss)

            if curr_loss < loss_min:
                loss_min = curr_loss
                self.log.info('New minmum loss, saving...')
                self._save_model(self.base_dir, 'best.model')
            with open(os.path.join(self.base_dir, 'losses.csv'), 'a') as fid:
                write = csv.writer(fid)
                write.writerow([epoch, train_loss, curr_loss])

        self.log.info('Training finished')

        return
