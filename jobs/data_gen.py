#!/usr/bin/env python
import logging
import multiprocessing as mp
import os
import csv
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from data.io import Reader
from lsm.lsmtype import Policy
import data.data_generators as Gen


class DataGenJob:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(self.config['log']['name'])
        self.log.info('Running Data Generator Job')
        self.output_dir = os.path.join(
            config['io']['data_dir'], config['data']['gen']['dir'])

    def _choose_generator(self) -> Gen.DataGenerator:
        choice = self.config['data']['gen']['generator']
        generators = {
            'TierCost': Gen.TierLevelGenerator(self.config, Policy.Tiering),
            'LevelCost': Gen.TierLevelGenerator(self.config, Policy.Leveling),
            'QCost': Gen.QCostGenerator(self.config),
            'KHybridCost': Gen.KHybridGenerator(self.config)}
        generator = generators.get(choice, None)
        if generator is None:
            self.log.error('Invalid generator choice. '
                           'Defaulting to KHybridCost')
            generator = Gen.KHybridGenerator(self.config)
        return generator

    def generate_csv_file(self, generator, idx: int, pos: int) -> int:
        fname_prefix = self.config['data']['gen']['file_prefix']
        fname = f'{fname_prefix}-{idx:04}.csv'
        fpath = os.path.join(self.output_dir, fname)
        if os.path.exists(fpath):
            self.log.info(f'{fpath} exists, exiting.')
            return -1

        samples = range(int(self.config['data']['gen']['samples']))
        header = generator.generate_header()
        with open(fpath, 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(header)
            for _ in tqdm(samples, desc=fname, position=pos, ncols=80):
                row = generator.generate_row()
                writer.writerow(row)

        return idx

    def generate_parquet_file(
            self,
            generator: Gen.DataGenerator,
            idx: int,
            pos: int) -> int:
        fname_prefix = self.config['data']['gen']['file_prefix']
        fname = f'{fname_prefix}-{idx:04}.parquet'
        fpath = os.path.join(self.output_dir, fname)
        if os.path.exists(fpath):
            self.log.info(f'{fpath} exists, exiting.')
            return -1

        samples = range(int(self.config['data']['gen']['samples']))
        table = []
        for _ in tqdm(samples, desc=fname, position=pos, ncols=80):
            table.append(generator.generate_row_parquet())
        table = pa.Table.from_pylist(table)
        pq.write_table(table, fpath)

        return idx

    def generate_file(self, idx: int) -> int:
        pos = 0
        if len(mp.current_process()._identity) > 0:
            pos = mp.current_process()._identity[0] - 1
        generator = self._choose_generator()

        if self.config['data']['gen']['format'] == 'parquet':
            self.generate_parquet_file(generator, idx, pos)
        else:  # format == 'csv'
            self.generate_csv_file(generator, idx, pos)

        return idx

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f'Writing all files to {self.output_dir}')

        inputs = list(range(0, self.config['data']['gen']['num_files']))
        threads = self.config['data']['gen']['num_workers']
        if threads == -1:
            threads = mp.cpu_count()
        self.log.info(f'{threads=}')

        if threads == 1:
            self.generate_file(0)
        else:
            with mp.Pool(
                    threads,
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
