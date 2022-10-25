#!/usr/bin/env python
import os
import torch
import toml
import logging
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchdata.datapipes as DataPipe

from data.io import Reader
from model.kcost import KCostModel


class Trainer:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(self.config['log']['name'])

    def prep_training(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.log.info(f'Using device: {self.device}')

        self.model = self._build_model()
        self.model = self.model.to(self.device)
        self.optimizer = self._build_optimizer(self.model)
        self.train_data, self.test_data = self._build_data()
        self.train_len = self.test_len = 0
        self.loss_fn = torch.nn.MSELoss()
        self.mean = np.array(self.config['train']['mean_bias'], np.float32)
        self.std = np.array(self.config['train']['std_bias'], np.float32)

    def _build_model(self):
        choice = self.config['model']['arch']
        self.log.info(f'Building model: {choice}')
        models = {
            'KCostModel': KCostModel,
        }
        model = models.get(choice, None)
        if model is None:
            self.log.warn('Invalid model arch. Defaulting to KCostModel')
            model = KCostModel
        model = model(self.config)

        return model

    def _build_optimizer(self, model):
        # optimizer = torch.optim.SGD(
        #         model.parameters(),
        #         lr=self.config['train']['learning_rate'])
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['train']['learning_rate'])

        return optimizer

    def _process_row(self, row):
        labels = np.array(row[0:4], np.float32)
        features = np.array(row[4:], np.float32)

        # First 4 are h, z0, z1, w, q
        # TODO: Streamline this process
        continuous_data = features[0:5]
        continuous_data -= self.mean
        continuous_data /= self.std

        # Remaining will be T and Ks
        categorical_data = features[5:]
        features = np.concatenate((continuous_data, categorical_data))

        return {'label': labels, 'feature': features}

    def _build_train(self):
        train_dir = os.path.join(
                self.config['io']['data_dir'],
                self.config['train']['dir'])
        dp_train = (DataPipe
                    .iter
                    .FileLister(train_dir)
                    .filter(filter_fn=lambda fname: fname.endswith('.csv'))
                    .open_files(mode='rt')
                    .parse_csv(delimiter=',', skip_lines=1)
                    .map(self._process_row)
                    .sharding_filter())
        if self.config['train']['shuffle'] is True:
            dp_train = dp_train.shuffle()

        train = DataLoader(
                dp_train,
                batch_size=self.config['train']['batch_size'],
                drop_last=self.config['train']['drop_last'],
                # Unsure if needed but to be safe
                shuffle=self.config['train']['shuffle'],
                num_workers=0)

        return train

    def _build_test(self):
        test_dir = os.path.join(
                self.config['io']['data_dir'],
                self.config['test']['dir'])
        dp_test = (DataPipe
                   .iter
                   .FileLister(test_dir)
                   .filter(filter_fn=lambda fname: fname.endswith('.csv'))
                   .open_files(mode='rt')
                   .parse_csv(delimiter=',', skip_lines=1)
                   .map(self._process_row)
                   .sharding_filter())
        if self.config['test']['shuffle'] is True:
            dp_test = dp_test.shuffle()
        test = DataLoader(
                dp_test,
                batch_size=self.config['test']['batch_size'],
                drop_last=self.config['test']['drop_last'],
                shuffle=self.config['test']['shuffle'],
                num_workers=0)

        return test

    def _build_data(self):
        train = self._build_train()
        test = self._build_test()

        return train, test

    def _train_loop(self):
        self.model.train()
        if self.train_len == 0:
            pbar = tqdm(self.train_data, ncols=80)
        else:
            pbar = tqdm(self.train_data, ncols=80, total=self.train_len)
        for batch, data in enumerate(pbar):
            label = data['label'].to(self.device)
            input = data['feature'].to(self.device)
            pred = self.model(input)
            loss = self.loss_fn(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % (1000) == 0:
                pbar.set_description(f'loss {loss:>5f}')

        if self.train_len == 0:
            self.train_len = batch + 1

    def _test_loop(self):
        self.model.eval()
        test_loss = 0
        if self.test_len == 0:
            pbar = tqdm(self.test_data, desc='testing', ncols=80)
        else:
            pbar = tqdm(self.test_data, desc='testing',
                        ncols=80, total=self.test_len)
        with torch.no_grad():
            for batch, data in enumerate(pbar):
                label = data['label'].to(self.device)
                feature = data['feature'].to(self.device)
                pred = self.model(feature)
                test_loss += self.loss_fn(pred, label).item()

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

    def train(self):
        save_dir = os.path.join(self.config['io']['data_dir'],
                                self.config['model']['dir'])
        os.makedirs(save_dir, exist_ok=True)
        self._dumpconfig(save_dir)

        loss_min = float('inf')
        early_stop_num = self.config['train']['early_stop_num']
        epsilon = self.config['train']['epsilon']
        max_epochs = self.config['train']['max_epochs']
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


if __name__ == '__main__':
    config = Reader.read_config('endure.toml')

    logging.basicConfig(format=config['log']['format'],
                        datefmt=config['log']['datefmt'])

    log = logging.getLogger(config['log']['name'])
    log.setLevel(config['log']['level'])

    a = Trainer(config)
    a.prep_training()
    a.train()
