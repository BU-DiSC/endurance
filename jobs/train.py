#!/usr/bin/env python
import os
import torch
import logging

from torch.utils.data import DataLoader
import torch.optim as TorchOpt

import data.kcost_dataset as EndureData
from data.io import Reader
from model.kcost import KCostModel
from model.tierlevelcost import TierLevelCost
from model.trainer import Trainer
import model.losses as Losses


class TrainJob:
    def __init__(self, config):
        self._config = config
        self.log = logging.getLogger(self._config['log']['name'])
        self.log.info('Running Training Job')
        self._dp = EndureData.EndureDataPipeGenerator(self._config)

    def _build_loss_fn(self) -> torch.nn.Module:
        losses = {
                'MSLE': Losses.MSLELoss(),
                'NMSE': Losses.NMSELoss(),
                'RMSLE': Losses.RMSLELoss(),
                'RMSE': Losses.RMSELoss(),
                'MSE': torch.nn.MSELoss(), }
        choice = self._config['train']['loss_fn']
        self.log.info(f'Loss function: {choice}')

        loss = losses.get(choice, None)
        if loss is None:
            self.log.warn('Invalid loss func. Defaulting to MSE')
            loss = loss.get('MSE')

        return loss

    def _build_model(self) -> torch.nn.Module:
        models = {
                'QCost': KCostModel,
                'TierLevelCost': TierLevelCost,
                'KCost': KCostModel, }
        choice = self._config['model']['arch']
        self.log.info(f'Building model: {choice}')
        model = models.get(choice, None)
        if model is None:
            self.log.warn('Invalid model arch. Defaulting to KCostModel')
            model = models.get('KCost')
        model = model(self._config)

        return model

    def _build_adam(self, model) -> TorchOpt.Adam:
        return TorchOpt.Adam(
                model.parameters(),
                lr=self._config['train']['learning_rate'],)

    def _build_adagrad(self, model) -> TorchOpt.Adagrad:
        return TorchOpt.Adagrad(
                model.parameters(),
                lr=self._config['train']['learning_rate'],)

    def _build_sgd(self, model) -> TorchOpt.SGD:
        return TorchOpt.SGD(model.parameters(),
                            lr=self._config['train']['learning_rate'],)

    def _build_optimizer(self, model) -> TorchOpt.Optimizer:
        optimizers = {
            'Adam': self._build_adam,
            'Adagrad': self._build_adagrad,
            'SGD': self._build_sgd}
        choice = self._config['train']['optimizer']
        self.log.info(f'Using optimizer : {choice}')
        opt_builder = optimizers.get(choice, None)
        if opt_builder is None:
            self.log.warn('Invalid optimizer choice, defaulting to SGD')
            opt_builder = optimizers.get('SGD')
        optimizer = opt_builder(model)

        return optimizer

    def _build_cosine_anneal(
            self,
            optimizer) -> TorchOpt.lr_scheduler.CosineAnnealingLR:
        return TorchOpt.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **self._config['train']['scheduler']['CosineAnnealingLR'],)

    def _build_scheduler(self, optimizer) -> TorchOpt.lr_scheduler._LRScheduler:
        schedules = {
                'CosineAnnealing': self._build_cosine_anneal,
                'None': None, }
        choice = self._config['train']['lr_scheduler']
        schedule_builder = schedules.get(choice, -1)
        if schedule_builder is -1:
            self.log.warn('Invalid scheduler, defaulting to none')
            return None
        if schedule_builder is None:
            return None
        scheduler = schedule_builder(optimizer)

        return scheduler

    def _build_train(self) -> DataLoader:
        train_dir = os.path.join(
                self._config['io']['data_dir'],
                self._config['train']['data']['dir'],)
        if self._config['train']['data']['use_dp']:
            train_data = self._dp.build_dp(
                    train_dir,
                    shuffle=self._config['train']['shuffle'],)
        else:
            train_data = EndureData.EndureIterableDataSet(
                    config=self._config,
                    folder=train_dir,
                    shuffle=self._config['train']['shuffle'],
                    format=self._config['train']['data']['format'],)
        train = DataLoader(
                train_data,
                batch_size=self._config['train']['batch_size'],
                drop_last=self._config['train']['drop_last'],
                num_workers=8,)

        return train

    def _build_test(self) -> DataLoader:
        test_dir = os.path.join(
                    self._config['io']['data_dir'],
                    self._config['test']['data']['dir'],)
        if self._config['test']['data']['use_dp']:
            test_data = self._dp.build_dp(
                    test_dir,
                    shuffle=self._config['test']['shuffle'],)
        else:
            test_data = EndureData.EndureIterableDataSet(
                    config=self._config,
                    folder=test_dir,
                    shuffle=self._config['test']['shuffle'],
                    format=self._config['test']['data']['format'],)
        test = DataLoader(
                test_data,
                batch_size=self._config['test']['batch_size'],
                drop_last=self._config['test']['drop_last'],
                num_workers=4,)

        return test

    def run(self) -> Trainer:
        model = self._build_model()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        train_data = self._build_train()
        test_data = self._build_test()
        loss_fn = self._build_loss_fn()

        trainer = Trainer(
                config=self._config,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_data=train_data,
                test_data=test_data,
                scheduler=scheduler,)
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
