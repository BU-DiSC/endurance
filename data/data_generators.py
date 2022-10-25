import logging
import numpy as np
import random
from itertools import combinations_with_replacement

import lsm.cost as CostFunc
from lsm.lsmtype import Policy


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger('endure')
        self._header = None

    def _sample_size_ratio(self) -> int:
        return np.random.randint(low=self.config['lsm']['size_ratio']['min'],
                                 high=self.config['lsm']['size_ratio']['max'])

    def _sample_bloom_filter_bits(self) -> float:
        sample = np.random.rand() * self.config['lsm']['bits_per_elem']['max']
        return np.around(sample, self.config['data_gen']['precision'])

    def _sample_workload(self, dimensions: int) -> list:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629/random-numbers-that-add-to-100-matlab
        workload = list(np.random.rand(dimensions - 1)) + [0, 1]
        workload.sort()

        return [b - a for a, b in zip(workload, workload[1:])]

    def generate_header(self):
        pass

    def generate_row(self):
        pass


class TierLevelGenerator(DataGenerator):
    def __init__(self, config, policy):
        super(TierLevelGenerator, self).__init__(config)
        self.policy = Policy.Tiering
        self.cf = CostFunc.EndureTierLevelCost(**self.config['system'])

    def generate_header(self):
        header = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost',
                  'h', 'z0', 'z1', 'q', 'w', 'T']
        return header

    def generate_row(self):
        z0, z1, q, w = self._sample_workload(4)
        T = self._sample_size_ratio()
        h = self._sample_bloom_filter_bits()

        line = [z0 * self.cf.Z0(h, T, self.policy),
                z1 * self.cf.Z1(h, T, self.policy),
                q * self.cf.Q(h, T, self.policy),
                w * self.cf.W(h, T, self.policy),
                h, z0, z1, q, w, T]
        return line


class KHybridGenerator(DataGenerator):
    def __init__(self, config):
        super(KHybridGenerator, self).__init__(config)
        self.cf = CostFunc.EndureKHybridCost(**self.config['system'])

    def _gen_k_levels(self, levels: int, max_size_ratio: int) -> list:
        arr = combinations_with_replacement(
            range(max_size_ratio, 0, -1), levels)

        return list(arr)

    def generate_header(self):
        header = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost',
                  'h', 'z0', 'z1', 'q', 'w', 'T']
        header += [f'K_{i}' for i in range(self.config['lsm']['max_levels'])]
        return header

    def generate_row(self):
        z0, z1, q, w = self._sample_workload(4)
        T = self._sample_size_ratio()
        h = self._sample_bloom_filter_bits()
        levels = int(self.cf.L(h, T, True))
        K = random.sample(self._gen_k_levels(levels, T - 1), 1)[0]
        K = np.pad(K, (0, self.config['lsm']['max_levels'] - len(K)))

        line = [z0 * self.cf.Z0(h, T, K), z1 * self.cf.Z1(h, T, K),
                q * self.cf.Q(h, T, K), w * self.cf.W(h, T, K),
                h, z0, z1, q, w, T]
        for level_idx in range(self.config['lsm']['max_levels']):
            line.append(K[level_idx])

        return line
