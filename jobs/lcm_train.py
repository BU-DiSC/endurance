#!/usr/bin/env python
import os
import torch
import logging

from torch.utils.data import DataLoader
import torch.optim as TorchOpt

from endure.data.io import Reader
from endure.lcm.data.classic_dataset import LCMDataPipeGenerator
from endure.lcm.data.iterable_dataset import LCMIterableDataSet
from endure.lcm.model.builder import LearnedCostModelBuilder
from endure.util.trainer import Trainer
import endure.util.losses as Losses


class LCMTrainJob:
    def __init__(self, config):
        self._config = config
        self.log = logging.getLogger(self._config['log']['name'])
        self.log.info('Running Training Job')
        self._dp = LCMDataPipeGenerator(self._config)
        self._model_builder = LearnedCostModelBuilder(self._config)

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
        return self._model_builder.build_model()

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

    def _build_scheduler(
            self,
            optimizer) -> TorchOpt.lr_scheduler._LRScheduler:
        schedules = {
                'CosineAnnealing': self._build_cosine_anneal,
                'None': None, }
        choice = self._config['train']['lr_scheduler']
        schedule_builder = schedules.get(choice, -1)
        if schedule_builder == -1:
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
            train_data = LCMIterableDataSet(
                config=self._config,
                folder=train_dir,
                shuffle=self._config['train']['data']['shuffle'],
                format=self._config['train']['data']['format'],)
        train = DataLoader(
            train_data,
            batch_size=self._config['train']['data']['batch_size'],
            drop_last=self._config['train']['data']['drop_last'],
            num_workers=self._config['train']['data']['num_workers'],)

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
            test_data = LCMIterableDataSet(
                config=self._config,
                folder=test_dir,
                shuffle=self._config['test']['data']['shuffle'],
                format=self._config['test']['data']['format'],)
        test = DataLoader(
            test_data,
            batch_size=self._config['test']['data']['batch_size'],
            drop_last=self._config['test']['data']['drop_last'],
            num_workers=self._config['train']['data']['num_workers'],)

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

    a = LCMTrainJob(config)
    a.run()
