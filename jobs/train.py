#!/usr/bin/env python
import os
import torch
import logging
import numpy as np

from torch.utils.data import DataLoader
import torchdata.datapipes as DataPipe

from data.io import Reader
from model.kcost import KCostModel
from model.tierlevelcost import TierLevelCost
from model.trainer import Trainer


class TrainJob:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(self.config['log']['name'])
        self.log.info('Running Training Job')
        self.prep_training()

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
            'TierLevelCost': TierLevelCost,
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
        categorical_data[0] -= self.config['lsm']['size_ratio']['min']
        features = np.concatenate((continuous_data, categorical_data))

        return (labels, features)

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
                    .in_memory_cache(size=2*8192)
                    .sharding_filter())
        if self.config['train']['shuffle'] is True:
            dp_train = dp_train.shuffle()
        train = DataLoader(
                dp_train,
                batch_size=self.config['train']['batch_size'],
                drop_last=self.config['train']['drop_last'],
                # Unsure if needed but to be safe
                shuffle=self.config['train']['shuffle'],
                num_workers=4)
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
                shuffle=self.config['test']['shuffle'])
        return test

    def _build_data(self):
        train = self._build_train()
        test = self._build_test()

        return train, test

    def run(self) -> Trainer:
        trainer = Trainer(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            train_data=self.train_data,
            test_data=self.test_data)
        trainer.run()

        return trainer


if __name__ == '__main__':
    config = Reader.read_config('endure.toml')

    logging.basicConfig(format=config['log']['format'],
                        datefmt=config['log']['datefmt'])

    log = logging.getLogger(config['log']['name'])
    log.setLevel(config['log']['level'])

    a = TrainJob(config)
    a.run()
