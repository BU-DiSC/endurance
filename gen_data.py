#!/usr/bin/env python
import logging
import multiprocessing as mp
import numpy as np
import os
import random
import csv
from tqdm import tqdm
from itertools import combinations_with_replacement

from data.io import Reader
import lsm.cost as CostFunc
from lsm.lsmtype import Policy


class DataGenerator:
    def __init__(self, config):
        self.cfg = config
        self.log = logging.getLogger('endure')
        self.reader = Reader(config)
        self.output_dir = os.path.join(
            config['io']['data_dir'], config['data_gen']['dir'])

    def _gen_cost_function(self) -> object:
        system = self.cfg['system']
        cost_function_choice = self.cfg['data_gen']['cost_function']
        cost_functions = {
            'EndureTierCost': CostFunc.EndureTierLevelCost(**system),
            'EndureLevelCost': CostFunc.EndureTierLevelCost(**system),
            'EndureKHybridCost': CostFunc.EndureKHybridCost(**system)}
        cf = cost_functions.get(cost_function_choice, None)
        if cf is None:
            self.log.error('Invalid cost function choice. '
                           'Defaulting to EndureKHybridCost')
            cf = CostFunc.EndureKHybridCost(**system)
        return cf

    def _gen_workload(self, dimensions: int) -> list:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629/random-numbers-that-add-to-100-matlab
        workload = list(np.random.rand(dimensions - 1)) + [0, 1]
        workload.sort()

        return [b - a for a, b in zip(workload, workload[1:])]

    def _gen_k_levels(self, levels: int, max_size_ratio: int) -> list:
        arr = combinations_with_replacement(
            range(max_size_ratio, 0, -1), levels)

        return list(arr)

    def _sample_size_ratio(self) -> int:
        return np.random.randint(low=self.cfg['lsm']['size_ratio']['min'],
                                 high=self.cfg['lsm']['size_ratio']['max'])

    def _sample_bloom_filter_bits(self) -> float:
        sample = np.random.rand() * self.cfg['lsm']['bits_per_elem']['max']
        return np.around(sample, self.cfg['data_gen']['precision'])

    def _create_endure_k_header(self):
        header = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost',
                  'h', 'z0', 'z1', 'q', 'w', 'T']
        header += [f'K_{i}' for i in range(self.cfg['lsm']['max_levels'])]
        return header

    def _create_endure_k_row(self, cf):
        z0, z1, q, w = self._gen_workload(4)
        T = self._sample_size_ratio()
        h = self._sample_bloom_filter_bits()
        levels = int(cf.L(h, T, True))
        K = random.sample(self._gen_k_levels(levels, T - 1), 1)[0]
        K = np.pad(K, (0, self.cfg['lsm']['max_levels'] - len(K)))

        line = [z0 * cf.Z0(h, T, K), z1 * cf.Z1(h, T, K),
                q * cf.Q(h, T, K), w * cf.W(h, T, K),
                h, z0, z1, q, w, T]
        for level_idx in range(self.cfg['lsm']['max_levels']):
            line.append(K[level_idx])

        return line

    def _create_endure_tier_level_header(self):
        return ['z0_cost', 'z1_cost', 'q_cost', 'w_cost',
                'h', 'z0', 'z1', 'q', 'w', 'T']

    def _create_endure_tier_level_row(self, cf, policy: Policy):
        z0, z1, q, w = self._gen_workload(4)
        T = self._sample_size_ratio()
        h = self._sample_bloom_filter_bits()

        line = [z0 * cf.Z0(h, T, policy), z1 * cf.Z1(h, T, policy),
                q * cf.Q(h, T, policy), w * cf.W(h, T, policy),
                h, z0, z1, q, w, T]

        return line

    def _create_endure_tier_row(self, cf):
        return self._create_endure_tier_level_row(cf, Policy.Tiering)

    def _create_endure_level_row(self, cf):
        return self._create_endure_tier_level_row(cf, Policy.Leveling)

    def gen_header(self) -> list:
        choices = {'EndureTierCost': self._create_endure_tier_level_header,
                   'EndureLevelCost': self._create_endure_tier_level_header,
                   'EndureKHybridCost': self._create_endure_k_header}
        cf_choice = self.cfg['data_gen']['cost_function']
        return choices.get(cf_choice)()

    def gen_row(self, cf) -> list:
        choices = {'EndureTierCost': self._create_endure_tier_row,
                   'EndureLevelCost': self._create_endure_level_row,
                   'EndureKHybridCost': self._create_endure_k_row}
        cf_choice = self.cfg['data_gen']['cost_function']
        return choices.get(cf_choice)(cf)

    def gen_file(self, idx: int) -> int:
        pos = 0
        if len(mp.current_process()._identity) > 0:
            pos = mp.current_process()._identity[0] - 1
        cf = self._gen_cost_function()
        fname_prefix = self.cfg['data_gen']['file_prefix']
        fname = f'{fname_prefix}-{idx:04}.csv'
        header = self.gen_header()

        samples = range(int(self.cfg['data_gen']['samples']))
        with open(os.path.join(self.output_dir, fname), 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(header)
            for _ in tqdm(samples, desc=fname, position=pos, ncols=80):
                row = self.gen_row(cf)
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

        with mp.Pool(threads,
                     initializer=tqdm.set_lock,
                     initargs=(mp.RLock(),)) as p:
            p.map(self.gen_file, inputs)

        return


if __name__ == '__main__':
    config = Reader.read_config('endure.toml')

    logging.basicConfig(format=config['log']['format'],
                        datefmt=config['log']['datefmt'])
    log = logging.getLogger('endure')
    log.setLevel(config['log']['level'])

    a = DataGenerator(config)
    a.run()
