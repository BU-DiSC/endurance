import os
import toml
import logging
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, config, model, optimizer, loss_fn, train_data,
                 test_data):
        self.config = config
        self.log = logging.getLogger(self.config['log']['name'])
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data = test_data
        self.train_len = self.test_len = 0
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.log.info(f'Training on device: {self.device}')

    def _train_step(self, label, features) -> float:
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

        for batch, (labels, features) in enumerate(pbar):
            labels = labels.to(self.device)
            features = features.to(self.device)
            loss = self._train_step(labels, features)
            if batch % (10) == 0:
                pbar.set_description(f'loss {loss:>5f}')

        if self.train_len == 0:
            self.train_len = batch + 1

        return loss

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
        torch.save(save_pt, os.path.join(save_dir, 'checkpoint.pt'))

    def _save_model(self, save_dir):
        torch.save(self.model.state_dict(),
                   os.path.join(save_dir, 'kcost_min.model'))

    def run(self):
        early_stop_num = self.config['train']['early_stop_num']
        epsilon = self.config['train']['epsilon']
        max_epochs = self.config['train']['max_epochs']
        save_dir = os.path.join(self.config['io']['data_dir'],
                                self.config['model']['dir'])

        os.makedirs(save_dir, exist_ok=True)
        self._dumpconfig(save_dir)

        loss_min = float('inf')
        losses = [float('inf')] * (early_stop_num + 1)
        self.log.info('Model parameters')
        for key in self.config['model'].keys():
            self.log.info(f'{key} = {self.config["model"][key]}')
        self.log.info('Training parameters')
        for key in self.config['train'].keys():
            self.log.info(f'{key} = {self.config["train"][key]}')

        for epoch in range(max_epochs):
            self.log.info(f'Epoch ({epoch+1}/{max_epochs})')
            self._train_loop()
            curr_loss = self._test_loop()
            self.log.info(f'Test loss: {curr_loss}')
            self._checkpoint(save_dir, epoch, curr_loss)
            if curr_loss < loss_min:
                loss_min = curr_loss
                self.log.info('New minmum loss, saving...')
                self._save_model(save_dir)

            losses.pop(0)
            losses.append(curr_loss)
            loss_deltas = [y - x for x, y in zip(losses, losses[1:])]
            self.log.info(f'Past losses ({losses})')
            if any([(x < epsilon and x > -epsilon) for x in loss_deltas]):
                self.log.info(f'Loss has only changed by {epsilon} for '
                              f'{early_stop_num} epochs. Terminating...')
                break

        self.log.info('Training finished')
