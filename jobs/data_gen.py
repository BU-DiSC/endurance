#!/usr/bin/env python
import logging
import multiprocessing as mp
import os
import csv
from tqdm import tqdm

from data.io import Reader
from lsm.lsmtype import Policy
import data.data_generators as Gen


class DataGenJob:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(self.config['log']['name'])
        self.log.info('Running Data Generator Job')
        self.reader = Reader(config)
        self.output_dir = os.path.join(
            config['io']['data_dir'], config['data_gen']['dir'])

    def _choose_generator(self):
        choice = self.config['data_gen']['generator']
        generators = {
            'TierCost': Gen.TierLevelGenerator(self.config, Policy.Tiering),
            'LevelCost': Gen.TierLevelGenerator(self.config, Policy.Leveling),
            'KHybridCost': Gen.KHybridGenerator(self.config)}
        generator = generators.get(choice, None)
        if generator is None:
            self.log.error('Invalid generator choice. '
                           'Defaulting to KHybridCost')
            generator = Gen.KHybridGenerator(self.config)
        return generator

    def generate_file(self, idx: int) -> int:
        pos = 0
        if len(mp.current_process()._identity) > 0:
            pos = mp.current_process()._identity[0] - 1
        generator = self._choose_generator()

        fname_prefix = self.config['data_gen']['file_prefix']
        fname = f'{fname_prefix}-{idx:04}.csv'
        header = generator.generate_header()

        samples = range(int(self.config['data_gen']['samples']))
        with open(os.path.join(self.output_dir, fname), 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(header)
            for _ in tqdm(samples, desc=fname, position=pos, ncols=80):
                row = generator.generate_row()
                writer.writerow(row)

        return idx

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f'Writing all files to {self.output_dir}')

        inputs = list(range(0, self.config['data_gen']['num_files']))
        threads = self.config['data_gen']['num_workers']
        if threads == -1:
            threads = mp.cpu_count()
        self.log.info(f'{threads=}')

        with mp.Pool(threads,
                     initializer=tqdm.set_lock,
                     initargs=(mp.RLock(),)) as p:
            p.map(self.generate_file, inputs)

        return


if __name__ == '__main__':
    config = Reader.read_config('endure.toml')

    logging.basicConfig(format=config['log']['format'],
                        datefmt=config['log']['datefmt'])
    log = logging.getLogger('endure')
    log.setLevel(config['log']['level'])

    a = DataGenJob(config)
    a.run()
