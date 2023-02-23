#!/usr/bin/env python
import csv
import logging
import multiprocessing as mp
import os
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from endure.data.io import Reader
from endure.ltune.data.generator import LTunerGenerator


class LTuneDataGenJob:
    def __init__(self, config):
        self.log = logging.getLogger(config['log']['name'])
        self.config = config
        self.setting = config['job']['LTuneDataGen']
        self.output_dir = os.path.join(
            self.config['io']['data_dir'], self.setting['dir'])

    def _choose_generator(self):
        return LTunerGenerator(self.config)

    def generate_csv_file(self, generator, idx: int, pos: int) -> int:
        fname_prefix = self.setting['file_prefix']
        fname = f'{fname_prefix}-{idx:04}.csv'
        fpath = os.path.join(self.output_dir, fname)
        early_exit = (os.path.exists(fpath)
                      and not self.setting['overwrite_if_exists'])
        if early_exit:
            self.log.info(f'{fpath} exists, exiting.')
            return -1

        samples = range(int(self.setting['samples']))
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
        generator: LTunerGenerator,
        idx: int,
        pos: int
    ) -> int:
        fname_prefix = self.setting['file_prefix']
        fname = f'{fname_prefix}-{idx:04}.parquet'
        fpath = os.path.join(self.output_dir, fname)
        early_exit = (os.path.exists(fpath)
                      and not self.setting['overwrite_if_exists'])
        if early_exit:
            self.log.info(f'{fpath} exists, exiting.')
            return -1

        samples = range(int(self.setting['samples']))
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

        if self.setting['format'] == 'parquet':
            self.generate_parquet_file(generator, idx, pos)
        else:  # format == 'csv'
            self.generate_csv_file(generator, idx, pos)

        return idx

    def run(self) -> None:
        self.log.info('Creating workload data')
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f'Writing all files to {self.output_dir}')

        inputs = list(range(0, self.setting['num_files']))
        threads = self.setting['num_workers']
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
    log = logging.getLogger(config['log']['name'])
    log.setLevel(config['log']['level'])

    a = LTuneDataGenJob(config)
    a.run()
