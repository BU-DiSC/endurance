import logging
import random
from typing import Any, List, Union, Optional
from itertools import combinations_with_replacement

import numpy as np

from endure.lsm.types import LSMDesign, System, Policy
from endure.lsm.cost import EndureCost
from endure.lcm.data.input_features import (
    kWORKLOAD_HEADER,
    kSYSTEM_HEADER,
    kCOST_HEADER,
    kINPUT_FEATS_DICT,
)


class LCMDataGenerator:
    def __init__(self, config: dict[str, Any], precision=3) -> None:
        self.log = logging.getLogger(config["log"]["name"])
        self._config = config
        self._header = None
        self.precision = precision
        self.cf = EndureCost(config)

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
        workload = np.around(np.random.rand(dimensions - 1), self.precision)
        workload = np.concatenate((workload, np.array([0, 1])))
        workload = np.sort(workload)

        return [b - a for a, b in zip(workload, workload[1:])]

    # TODO: Will want to configure environment to simulate larger ranges over
    # potential system values
    def _sample_entry_per_page(self, entry_size: int = 8192) -> int:
        # Potential page sizes are 4KB, 8KB, 16KB
        KB_TO_BITS = 8 * 1024
        page_sizes = np.array(self._config["generator"]["page_sizes"])
        entries_per_page = (page_sizes * KB_TO_BITS) / entry_size
        return np.random.choice(entries_per_page)

    def _sample_selectivity(self) -> float:
        low, high = self._config["generator"]["selectivity_range"]
        return (high - low) * np.random.rand() + low

    def _sample_entry_size(self) -> int:
        choices = self._config["generator"]["entry_sizes"]
        return np.random.choice(choices)

    def _sample_memory_budget(self) -> float:
        low, high = self._config["generator"]["memory_budget"]
        return (high - low) * np.random.rand() + low

    def _sample_total_elements(self) -> int:
        low, high = self._config["generator"]["elements_range"]
        return np.random.randint(low=low, high=high)

    def _sample_system(self) -> System:
        E = self._sample_entry_size()
        B = self._sample_entry_per_page(entry_size=E)
        s = self._sample_selectivity()
        H = self._sample_memory_budget()
        N = self._sample_total_elements()
        system = System(E, s, B, N, H)

        return system

    def _sample_design(
        self,
        system: System,
        policies: List[Policy] = [Policy.Tiering, Policy.Leveling],
    ) -> LSMDesign:
        EPSILON = 0.1
        h = self._sample_bloom_filter_bits(max=(system.H - EPSILON))
        T = self._sample_size_ratio()
        policy = policies[0]
        if len(policies) > 1:
            policy = random.choice(policies)
        lsm = LSMDesign(h, T, policy)

        return lsm

    def _sample_config(self) -> tuple:
        system = self._sample_system()
        design = self._sample_design(system)

        return (system.B, system.s, system.E, system.H, system.N, design.h, design.T)

    def _gen_system_header(self) -> list:
        return kSYSTEM_HEADER

    def _gen_workload_header(self) -> list:
        return kWORKLOAD_HEADER

    def _gen_cost_header(self) -> list:
        return kCOST_HEADER

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

    def generate_row(
        self, row_type: str = "parquet"
    ) -> Union[list, dict[str, Union[int, float]]]:
        if row_type == "parquet":
            row = self.generate_row_parquet()
        else:  # format == 'csv'
            row = self.generate_row_csv()

        return row


class ClassicGenerator(LCMDataGenerator):
    def __init__(
        self,
        config: dict[str, Any],
        precision: int = 3,
        policies: List[Policy] = [Policy.Tiering, Policy.Leveling],
    ):
        super().__init__(config, precision)
        self.policies = policies
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["policy", "h", "T"]
        self.header = cost_header + workload_header + system_header + decision

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        system: System = self._sample_system()
        design: LSMDesign = self._sample_design(system, policies=self.policies)

        line = [
            z0 * self.cf.Z0(design, system),
            z1 * self.cf.Z1(design, system),
            q * self.cf.Q(design, system),
            w * self.cf.W(design, system),
            z0,
            z1,
            q,
            w,
            system.B,
            system.s,
            system.E,
            system.H,
            system.N,
            0 if design.policy == Policy.Tiering else 1,
            design.h,
            design.T,
        ]

        return line


class KHybridGenerator(LCMDataGenerator):
    def __init__(self, config: dict[str, Any], precision: int = 3):
        super().__init__(config, precision)
        self.max_levels = self._config["lsm"]["max_levels"]
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T"]
        self.header = cost_header + workload_header + system_header + decision
        self.header += [f"K_{i}" for i in range(self.max_levels)]

    def _sample_design(self, system: System) -> LSMDesign:
        design = super()._sample_design(system)
        h = design.h
        T = design.T
        levels = int(self.cf.L(design, system, ceil=True))
        k = np.random.randint(low=1, high=int(T), size=(levels))
        # k = random.sample(self._gen_k_levels(levels, int(T) - 1), 1)[0]
        remaining = np.ones(self.max_levels - len(k))
        k = np.concatenate([k, remaining])
        design = LSMDesign(h=h, T=T, policy=Policy.KHybrid, K=k.tolist())

        return design

    def _gen_k_levels(self, levels: int, max_size_ratio: int) -> list:
        arr = combinations_with_replacement(range(max_size_ratio, 0, -1), levels)

        return list(arr)

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        system: System = self._sample_system()
        design: LSMDesign = self._sample_design(system)

        line = [
            z0 * self.cf.Z0(design, system),
            z1 * self.cf.Z1(design, system),
            q * self.cf.Q(design, system),
            w * self.cf.W(design, system),
            z0,
            z1,
            q,
            w,
            system.B,
            system.s,
            system.E,
            system.H,
            system.N,
            design.h,
            design.T,
        ]
        for level_idx in range(self.max_levels):
            line.append(design.K[level_idx])

        return line


class QCostGenerator(LCMDataGenerator):
    def __init__(self, config: dict[str, Any], precision: int = 3):
        super().__init__(config, precision)
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T", "Q"]
        self.header = cost_header + workload_header + system_header + decision

    def _sample_q(self, max_size_ratio: int) -> int:
        return np.random.randint(
            low=self._config["lsm"]["size_ratio"]["min"] - 1,
            high=max_size_ratio,
        )

    def _sample_design(self, system: System) -> LSMDesign:
        design = super()._sample_design(system)
        h = design.h
        T = design.T
        Q = self._sample_q(int(T))
        design = LSMDesign(h=h, T=T, policy=Policy.QFixed, Q=Q)

        return design

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        system: System = self._sample_system()
        design: LSMDesign = self._sample_design(system)

        line = [
            z0 * self.cf.Z0(design, system),
            z1 * self.cf.Z1(design, system),
            q * self.cf.Q(design, system),
            w * self.cf.W(design, system),
            z0,
            z1,
            q,
            w,
            system.B,
            system.s,
            system.E,
            system.H,
            system.N,
            design.h,
            design.T,
            design.Q,
        ]
        return line


class QCostBinaryGenerator(QCostGenerator):
    def __init__(self, config: dict[str, Any], precision: int = 3):
        super().__init__(config, precision)

    def _sample_q(self, max_size_ratio: int) -> int:
        choices = (self._config["lsm"]["size_ratio"]["min"] - 1, max_size_ratio - 1)
        return np.random.choice(choices)


class YZCostGenerator(LCMDataGenerator):
    def __init__(self, config: dict[str, Any], precision: int = 3):
        super().__init__(config, precision)
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T", "Y", "Z"]
        self.header = cost_header + workload_header + system_header + decision

    def _sample_capacity(self, max_size_ratio: int) -> int:
        return np.random.randint(
            low=self._config["lsm"]["size_ratio"]["min"] - 1,
            high=max_size_ratio,
        )

    def _sample_design(self, system: System) -> LSMDesign:
        design = super()._sample_design(system)
        h = design.h
        T = design.T
        Y = self._sample_capacity(int(T))
        Z = self._sample_capacity(int(T))
        design = LSMDesign(h=h, T=T, policy=Policy.YZHybrid, Y=Y, Z=Z)

        return design

    def generate_header(self) -> list:
        return self.header

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        system: System = self._sample_system()
        design: LSMDesign = self._sample_design(system)

        line = [
            z0 * self.cf.Z0(design, system),
            z1 * self.cf.Z1(design, system),
            q * self.cf.Q(design, system),
            w * self.cf.W(design, system),
            z0,
            z1,
            q,
            w,
            system.B,
            system.s,
            system.E,
            system.H,
            system.N,
            design.h,
            design.T,
            design.Y,
            design.Z,
        ]
        return line
