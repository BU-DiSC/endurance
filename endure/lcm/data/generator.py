import logging
import numpy as np
import random
from copy import deepcopy
from typing import Union, Optional
from itertools import combinations_with_replacement

import endure.lsm.cost_model as CostFunc


class LCMDataGenerator:
    def __init__(self, config):
        self.log = logging.getLogger(config["log"]["name"])
        self._config = config
        self._header = None

    def _sample_size_ratio(self) -> int:
        return np.random.randint(
            low=self._config["lsm"]["size_ratio"]["min"],
            high=self._config["lsm"]["size_ratio"]["max"],
        )

    def _sample_bloom_filter_bits(self, max: Optional[float] = None) -> float:
        if max is None:
            max = self._config["lsm"]["bits_per_elem"]["max"]
        min = self._config["lsm"]["bits_per_elem"]["min"]
        sample = (max - min) * np.random.rand() + min
        return np.around(sample, self._config["lcm"]["data"]["precision"])

    def _sample_workload(self, dimensions: int) -> list:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629
        workload = list(np.random.rand(dimensions - 1)) + [0, 1]
        workload.sort()

        return [b - a for a, b in zip(workload, workload[1:])]

    # TODO: Will want to configure environment to simulate larger ranges over
    # potential system values
    def _sample_entry_per_page(self, entry_size: int = 8192) -> int:
        # Potential page sizes are 4KB, 8KB, 16KB
        KB_TO_BITS = 8 * 1024
        entries_per_page = (np.array([4, 8, 16]) * KB_TO_BITS) / entry_size
        return np.random.choice(entries_per_page)

    def _sample_selectivity(self) -> float:
        low, high = (1e-6, 1e-7)
        return (high - low) * np.random.rand() + low

    def _sample_entry_size(self) -> int:
        return np.random.choice([1024, 2048, 4096, 8192])

    def _sample_memory_budget(self) -> float:
        low, high = (5, 20)
        return (high - low) * np.random.rand() + low

    def _sample_total_elements(self) -> int:
        return np.random.randint(low=100000000, high=1000000000)

    def _sample_system(self) -> tuple:
        E = self._sample_entry_size()
        B = self._sample_entry_per_page(entry_size=E)
        s = self._sample_selectivity()
        H = self._sample_memory_budget()
        N = self._sample_total_elements()

        return (B, s, E, H, N)

    def _sample_config(self) -> tuple:
        EPSILON = 0.1
        (B, s, E, H, N) = self._sample_system()
        h = self._sample_bloom_filter_bits(max=(H - EPSILON))
        T = self._sample_size_ratio()

        return (B, s, E, H, N, h, T)

    def _gen_system_header(self) -> list:
        return ["B", "s", "E", "H", "N"]

    def _gen_workload_header(self) -> list:
        return ["z0", "z1", "q", "w"]

    def _gen_cost_header(self) -> list:
        return ["z0_cost", "z1_cost", "q_cost", "w_cost"]

    def generate_row_csv(self) -> list:
        return []

    def generate_row_parquet(self) -> dict:
        header = self.generate_header()
        row = self.generate_row_csv()
        line = {}
        for key, val in zip(header, row):
            line[key] = val

        return line

    def generate_header(self) -> list:
        return []

    def generate_row(self) -> Union[list, dict]:
        if self._config["data"]["gen"]["format"] == "parquet":
            row = self.generate_row_parquet()
        else:  # format == 'csv'
            row = self.generate_row_csv()

        return row


class LevelGenerator(LCMDataGenerator):
    def __init__(self, config: dict):
        super().__init__(config)
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T"]
        self.header = cost_header + workload_header + system_header + decision

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        B, s, E, H, N, h, T = self._sample_config()

        config = deepcopy(self._config)
        config["lsm"]["system"]["B"] = B
        config["lsm"]["system"]["s"] = s
        config["lsm"]["system"]["E"] = E
        config["lsm"]["system"]["H"] = H
        config["lsm"]["system"]["N"] = N
        # TODO: Change cost function to dynamically take in system var
        cf = CostFunc.EndureLevelCost(config)

        line = [
            z0 * cf.Z0(h, T),
            z1 * cf.Z1(h, T),
            q * cf.Q(h, T),
            w * cf.W(h, T),
            z0,
            z1,
            q,
            w,
            B,
            s,
            E,
            H,
            N,
            h,
            T,
        ]
        return line


class TierGenerator(LCMDataGenerator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.cf = CostFunc.EndureTierCost(config)
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T"]
        self.header = cost_header + workload_header + system_header + decision

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        T = self._sample_size_ratio()
        h = self._sample_bloom_filter_bits()
        B = self._sample_entry_per_page()
        s = self._sample_selectivity()
        E = self._sample_entry_size()
        H = self._sample_memory_budget()
        N = self._sample_total_elements()

        line = [
            z0 * self.cf.Z0(h, T),
            z1 * self.cf.Z1(h, T),
            q * self.cf.Q(h, T),
            w * self.cf.W(h, T),
            z0,
            z1,
            q,
            w,
            B,
            s,
            E,
            H,
            N,
            h,
            T,
        ]
        return line


class KHybridGenerator(LCMDataGenerator):
    def __init__(self, config: dict):
        super(KHybridGenerator, self).__init__(config)
        self.cf = CostFunc.EndureKCost(self._config)
        max_levels = self._config["lsm"]["max_levels"]
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T"]
        self.header = cost_header + workload_header + system_header + decision
        self.header += [f"K_{i}" for i in range(max_levels)]

    def _gen_k_levels(self, levels: int, max_size_ratio: int) -> list:
        arr = combinations_with_replacement(range(max_size_ratio, 0, -1), levels)

        return list(arr)

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        T = self._sample_size_ratio()
        h = self._sample_bloom_filter_bits()
        B = self._sample_entry_per_page()
        s = self._sample_selectivity()
        E = self._sample_entry_size()
        H = self._sample_memory_budget()
        N = self._sample_total_elements()
        levels = int(self.cf.cf.L(h, T, True))
        K = random.sample(self._gen_k_levels(levels, T - 1), 1)[0]
        K = np.pad(K, (0, self._config["lsm"]["max_levels"] - len(K)))

        line = [
            z0 * self.cf.Z0(h, T, K),
            z1 * self.cf.Z1(h, T, K),
            q * self.cf.Q(h, T, K),
            w * self.cf.W(h, T, K),
            z0,
            z1,
            q,
            w,
            B,
            s,
            E,
            H,
            N,
            h,
            T,
        ]
        for level_idx in range(self._config["lsm"]["max_levels"]):
            line.append(K[level_idx])
        return line


class QCostGenerator(LCMDataGenerator):
    def __init__(self, config: dict):
        super(QCostGenerator, self).__init__(config)
        self.cf = CostFunc.EndureQCost(self._config)
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T", "Q"]
        self.header = cost_header + workload_header + system_header + decision

    def _sample_q(self) -> int:
        return np.random.randint(
            low=self._config["lsm"]["size_ratio"]["min"] - 1,
            high=self._config["lsm"]["size_ratio"]["max"] - 1,
        )

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        B = self._sample_entry_per_page()
        s = self._sample_selectivity()
        E = self._sample_entry_size()
        H = self._sample_memory_budget()
        N = self._sample_total_elements()
        T = self._sample_size_ratio()
        h = self._sample_bloom_filter_bits()
        Q = self._sample_q()

        line = [
            z0 * self.cf.Z0(h, T, Q),
            z1 * self.cf.Z1(h, T, Q),
            q * self.cf.Q(h, T, Q),
            w * self.cf.W(h, T, Q),
            z0,
            z1,
            q,
            w,
            B,
            s,
            E,
            H,
            N,
            h,
            T,
            Q,
        ]
        return line
