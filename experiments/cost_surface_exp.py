import logging
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from data.dataio import Writer
from lsm.lsmtype import LSMSystem
import lsm.cost as CostFunc

MAX_LEVELS = 16


class CostSurfaceExp():
    def __init__(self, config: dict) -> None:
        self.config = config
        self.log = logging.getLogger('endure')
        self.writer = Writer(self.config)

    def wl_to_array(self, wl_dict):
        return (wl_dict['id'], wl_dict['z0'], wl_dict['z1'], wl_dict['q'], wl_dict['w'])

    def calc_qcosts(self):
        system_vars = LSMSystem(**self.config['system'])
        q_cf = CostFunc.EndureQFixedCost(**self.config['system'])

        df = []
        for wl in self.config['inputs']['workloads']:
            wl_id, z0, z1, q, w = self.wl_to_array(wl)
            self.log.info(f'[QFixedCost] Workload: ({z0}, {z1}, {q}, {w})')
            for h in np.arange(0, system_vars.H, 0.5):
                for T in np.arange(2, 100, 4):
                    tier_cost = q_cf.calc_cost(h, T, T - 1, z0, z1, q, w)
                    level_cost = q_cf.calc_cost(h, T, 1, z0, z1, q, w)
                    for Q in range(1, T):
                        q_cost = q_cf.calc_cost(h, T, Q, z0, z1, q, w)
                        row = {'wl_id': wl_id,
                               'h': h,
                               'T': T,
                               'z0': z0,
                               'z1': z1,
                               'q': q,
                               'w': w,
                               'Q': Q,
                               'new_cost': q_cost,
                               'tier_cost': tier_cost,
                               'level_cost': level_cost,
                               'old_cost': min([tier_cost, level_cost]),
                               }
                        df.append(row)

        return pd.DataFrame(df)

    def calc_yzcost(self):
        system_vars = LSMSystem(**self.config['system'])
        cf = CostFunc.EndureYZHybridCost(**self.config['system'])

        df = []
        for wl in self.config['inputs']['workloads']:
            wl_id, z0, z1, q, w = self.wl_to_array(wl)
            self.log.info(f'[YZCost] Workload: ({z0}, {z1}, {q}, {w})')
            for h in np.arange(0, system_vars.H, 0.25):
                for T in np.arange(2, 50):
                    tier_cost = cf.calc_cost(h, T, T - 1, T - 1, z0, z1, q, w)
                    level_cost = cf.calc_cost(h, T, 1, 1, z0, z1, q, w)
                    for Y in range(1, T):
                        for Z in range(0, Y):
                            Z += 1
                            new_cost = cf.calc_cost(h, T, Y, Z, z0, z1, q, w)
                            row = {'wl_id': wl_id,
                                   'h': h,
                                   'T': T,
                                   'z0': z0,
                                   'z1': z1,
                                   'q': q,
                                   'w': w,
                                   'Y': Y,
                                   'Z': Z,
                                   'new_cost': new_cost,
                                   'tier_cost': tier_cost,
                                   'level_cost': level_cost,
                                   'old_cost': min([tier_cost, level_cost]),
                                   }
                            df.append(row)

        return pd.DataFrame(df)

    def calc_kcost(self):
        system_vars = LSMSystem(**self.config['system'])
        cf = CostFunc.EndureKHybridCost(**self.config['system'])

        for wl in self.config['inputs']['workloads']:
            wl_id, z0, z1, q, w = self.wl_to_array(wl)
            self.log.info(f'[KCost] Workload: ({z0}, {z1}, {q}, {w})')
            df = []
            for h in np.arange(0, system_vars.H, 0.5):
                for T in np.arange(2, 50, 2):
                    levels = int(cf.L(h, T, True))
                    level_assignments = self.create_k_levels(levels, T - 1)
                    tiering = np.array([T - 1] * levels)
                    leveling = np.array([1] * levels)
                    tier_cost = cf.calc_cost(h, T, tiering, z0, z1, q, w)
                    level_cost = cf.calc_cost(h, T, leveling, z0, z1, q, w)
                    for K in level_assignments:
                        K = np.pad(K, (0, MAX_LEVELS - len(K)))
                        new_cost = cf.calc_cost(h, T, K, z0, z1, q, w)
                        row = {'wl_id': wl_id,
                               'h': h,
                               'T': T,
                               'z0': z0,
                               'z1': z1,
                               'q': q,
                               'w': w,
                               'B': self.config['system']['B'],
                               'phi': self.config['system']['phi'],
                               's': self.config['system']['s'],
                               'E': self.config['system']['E'],
                               'H': self.config['system']['H'],
                               'N': self.config['system']['N'],
                               'new_cost': new_cost,
                               'tier_cost': tier_cost,
                               'level_cost': level_cost,
                               'old_cost': min([tier_cost, level_cost]),
                               }
                        for level_idx in range(MAX_LEVELS):
                            row[f'K_{level_idx}'] = K[level_idx]
                        df.append(row)
            df = pd.DataFrame(df)
            self.log.info(f'Writing workload ID {wl_id}')
            df.to_feather(f'k_wl_{wl_id}.feather')

        return pd.DataFrame(df)

    def create_k_levels(self, levels: int, max_size_ratio: int):
        arr = combinations_with_replacement(range(max_size_ratio, 0, -1), levels)

        return arr

    def run(self) -> None:
        self.log.info('Cost Surface Experiment')

        # arr = self.create_k_levels(5, 10)
        # for e in arr:
        #     self.log.info(e)

        df = self.calc_kcost()
        df.to_feather('cost_surface_k.feather')
        # self.writer.export_csv_file(df, 'cost_surface_k_cost.csv')

        # df = self.calc_qcosts()
        # self.writer.export_csv_file(df, 'cost_surface_q_cost.csv')

        # df = self.calc_yzcost()
        # self.writer.export_csv_file(df, 'cost_surface_yz_cost.csv')

        return None
