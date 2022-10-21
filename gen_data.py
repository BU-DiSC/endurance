#!/usr/bin/env python
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import random
import csv
from tqdm import tqdm
from itertools import combinations_with_replacement

from data.io import Reader
import lsm.cost as CostFunc


class DataGenerator:
    def __init__(self, config):
        self.cfg = config
        self.log = logging.getLogger('endure')
        self.reader = Reader(config)
        self.output_dir = os.path.join(
            config['io']['data_dir'], config['data_gen']['dir'])

    def _generate_cost_function(self) -> object:
        system = self.cfg['system']
        cost_function_choice = self.cfg['data_gen']['cost_function']
        cost_functions = {
            'EndureKHybridCost': CostFunc.EndureKHybridCost(**system),
        }
        cf = cost_functions.get(cost_function_choice, None)
        if cf is None:
            self.log.error('Invalid cost function choice. '
                           'Defaulting to EndureKHybridCost')
            cf = CostFunc.EndureKHybridCost(**system)

        return cf

    def gen_workload(self, dimensions: int) -> list:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629/random-numbers-that-add-to-100-matlab
        workload = list(np.random.rand(dimensions - 1)) + [0, 1]
        workload.sort()

        return [b - a for a, b in zip(workload, workload[1:])]

    def create_k_levels(self, levels: int, max_size_ratio: int) -> list:
        arr = combinations_with_replacement(
            range(max_size_ratio, 0, -1), levels)

        return list(arr)

    def create_row(self, cf, h, T, K, z0, z1, q, w) -> list:
        line = [
            z0 * cf.Z0(h, T, K),
            z1 * cf.Z1(h, T, K),
            q * cf.Q(h, T, K),
            w * cf.W(h, T, K),
            h,
            z0,
            z1,
            q,
            w,
            T,
        ]
        for level_idx in range(self.cfg['lsm']['max_levels']):
            line.append(K[level_idx])

        return line

    def gen_file(self, idx: int) -> int:
        cf = self._generate_cost_function()
        fname_prefix = self.cfg['data_gen']['file_prefix']
        fname = f'{fname_prefix}-{idx:04}.csv'
        header = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost',
                  'h', 'z0', 'z1', 'q', 'w', 'T']
        header += [f'K_{i}' for i in range(self.cfg['lsm']['max_levels'])]

        with open(os.path.join(self.output_dir, fname), 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(header)
            samples = range(int(self.cfg['data_gen']['samples']))
            pos = mp.current_process()._identity[0] - 1
            for _ in tqdm(samples, desc=fname, position=pos, ncols=80):
                z0, z1, q, w = self.gen_workload(4)
                T = np.random.randint(
                    low=self.cfg['lsm']['size_ratio']['min'],
                    high=self.cfg['lsm']['size_ratio']['max'])
                h = np.around(
                    self.cfg['lsm']['bits_per_elem']['max'] * np.random.rand(),
                    self.cfg['data_gen']['precision'])

                levels = int(cf.L(h, T, True))
                K = random.sample(self.create_k_levels(levels, T - 1), 1)[0]
                K = np.pad(K, (0, self.cfg['lsm']['max_levels'] - len(K)))
                row = self.create_row(cf, h, T, K, z0, z1, q, w)
                writer.writerow(row)

        return idx

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f'Writing all files to {self.output_dir}')

        inputs = list(range(0, self.cfg['data_gen']['num_files']))
        threads = self.cfg['data_gen']['num_workers']
        if threads == -1:
            threads = mp.cpu_count()
        self.log.info(f'{threads=}')

        with mp.Pool(
            threads, initializer=tqdm.set_lock, initargs=(mp.RLock(),)
        ) as p:
            p.map(self.gen_file, inputs)

        return


if __name__ == '__main__':
    config = Reader.read_config('endure.toml')

    logging.basicConfig(
        format=config['log']['format'],
        datefmt=config['log']['datefmt'])

    log = logging.getLogger('endure')
    log.setLevel(config['log']['level'])

    a = DataGenerator(config)
    a.run()
