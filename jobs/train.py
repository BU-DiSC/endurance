#!/usr/bin/env python
import os
import torch
import logging

from torch.utils.data import DataLoader

import data.kcost_dataset as EndureData
from data.io import Reader
from model.kcost import KCostModel
from model.tierlevelcost import TierLevelCost
from model.trainer import Trainer
from model.losses import MSLELoss


class TrainJob:
    def __init__(self, config):
        self._config = config
        self.log = logging.getLogger(self._config['log']['name'])
        self.log.info('Running Training Job')
        self._dp = EndureData.EndureDataPipeGenerator(self._config)

    def _build_model(self):
        choice = self._config['model']['arch']
        self.log.info(f'Building model: {choice}')
        models = {
            'KCost': KCostModel,
            'QCost': KCostModel,
            'TierLevelCost': TierLevelCost,
        }
        model = models.get(choice, None)
        if model is None:
            self.log.warn('Invalid model arch. Defaulting to KCostModel')
            model = models.get('KCost')
        model = model(self._config)

        return model

    def _build_optimizer(self, model):
        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     lr=self._config['train']['learning_rate'])
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self._config['train']['learning_rate'])

        return optimizer

    def _build_train(self):
        train_dir = os.path.join(
            self._config['io']['data_dir'],
            self._config['train']['dir'])
        if self._config['train']['use_dp']:
            train_data = self._dp.build_dp(
                train_dir,
                shuffle=self._config['train']['shuffle'])
        else:
            train_data = EndureData.EndureIterableDataSet(
                config=self._config,
                folder=train_dir,)
            # train_data = EndureData.EndureDataSet(
            #     config=self._config,
            #     folder=train_dir,)
        train = DataLoader(
            train_data,
            batch_size=self._config['train']['batch_size'],
            drop_last=self._config['train']['drop_last'],)
            # num_workers=4,
            # shuffle=self._config['train']['shuffle'])
        return train

    def _build_test(self):
        test_dir = os.path.join(
                self._config['io']['data_dir'],
                self._config['test']['dir'])
        if self._config['test']['use_dp']:
            test_data = self._dp.build_dp(
                test_dir,
                shuffle=self._config['test']['shuffle'])
        else:
            test_data = EndureData.EndureIterableDataSet(
                config=self._config,
                folder=test_dir,)
        test = DataLoader(
            test_data,
            batch_size=self._config['test']['batch_size'],
            drop_last=self._config['test']['drop_last'],)
            # shuffle=self._config['test']['shuffle'])
        return test

    def _build_data(self):
        train = self._build_train()
        test = self._build_test()

        return train, test

    def run(self) -> Trainer:
        model = self._build_model()
        optimizer = self._build_optimizer(model)
        train_data = self._build_train()
        test_data = self._build_test()
        loss_fn = MSLELoss()

        trainer = Trainer(
            config=self._config,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            test_data=test_data)
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
