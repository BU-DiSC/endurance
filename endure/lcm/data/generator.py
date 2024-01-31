import logging
import random
from typing import List, Optional, Tuple
from itertools import combinations_with_replacement

import numpy as np

from endure.lsm.types import LSMDesign, System, Policy
from endure.lsm.cost import EndureCost
from endure.lcm.data.input_features import (
    kWORKLOAD_HEADER,
    kSYSTEM_HEADER,
    kCOST_HEADER,
)


class LCMDataGenerator:
    def __init__(
        self,
        bits_per_elem_range: Tuple[int, int] = (1, 10),
        size_ratio_range: Tuple[int, int] = (2, 31),
        page_sizes: List[int] = [4, 8, 16],
        entry_sizes: List[int] = [1024, 2048, 4096, 8192],
        memory_budget_range: Tuple[float, float] = (5.0, 20.0),
        selectivity_range: Tuple[float, float] = (1e-7, 1e-9),
        elements_range: Tuple[int, int] = (100000000, 1000000000),
        max_levels: int = 16,
        precision: int = 3,
    ) -> None:
        self._header = None
        self.precision = precision

        self.bits_per_elem_min = bits_per_elem_range[0]
        self.bits_per_elem_max = bits_per_elem_range[1]
        self.size_ratio_min = size_ratio_range[0]
        self.size_ratio_max = size_ratio_range[1]
        self.entry_sizes = entry_sizes
        self.memory_budget_range = memory_budget_range
        self.page_sizes = page_sizes
        self.selectivity_range = selectivity_range
        self.elements_range = elements_range
        self.max_levels = max_levels
        self.cf = EndureCost(max_levels=max_levels)

        self.header = []

    def _sample_size_ratio(self) -> int:
        return np.random.randint(low=self.size_ratio_min, high=self.size_ratio_max)

    def _sample_bloom_filter_bits(self, max: Optional[float] = None) -> float:
        if max is None:
            max = self.bits_per_elem_max
        min = self.bits_per_elem_min
        sample = (max - min) * np.random.rand() + min
        return np.around(sample, self.precision)

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
        page_sizes = np.array(self.page_sizes)
        entries_per_page = (page_sizes * KB_TO_BITS) / entry_size
        return np.random.choice(entries_per_page)

    def _sample_selectivity(self) -> float:
        low, high = self.selectivity_range
        return (high - low) * np.random.rand() + low

    def _sample_entry_size(self) -> int:
        return np.random.choice(self.entry_sizes)

    def _sample_memory_budget(self) -> float:
        low, high = self.memory_budget_range
        return (high - low) * np.random.rand() + low

    def _sample_total_elements(self) -> int:
        low, high = self.elements_range
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
    ) -> LSMDesign:
        EPSILON = 0.1
        h = self._sample_bloom_filter_bits(max=(system.H - EPSILON))
        T = self._sample_size_ratio()
        lsm = LSMDesign(h, T)

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

    def _gen_row_data(self) -> list:
        return []

    def generate_header(self) -> list:
        return self.header

    def generate_row(self) -> dict:
        header = self.generate_header()
        row = self._gen_row_data()
        line = {}
        for key, val in zip(header, row):
            line[key] = val

        return line


class ClassicGenerator(LCMDataGenerator):
    def __init__(
        self,
        policies: List[Policy] = [Policy.Tiering, Policy.Leveling],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policies = policies
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["policy", "h", "T"]
        self.header = cost_header + workload_header + system_header + decision

    def _sample_design(
        self,
        system: System,
    ) -> LSMDesign:
        EPSILON = 0.1
        h = self._sample_bloom_filter_bits(max=(system.H - EPSILON))
        T = self._sample_size_ratio()
        policy = self.policies[0]
        if len(self.policies) > 1:
            policy = random.choice(self.policies)
        lsm = LSMDesign(h, T, policy)

        return lsm

    def _gen_row_data(self) -> list:
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
            0 if design.policy == Policy.Tiering else 1,
            design.h,
            design.T,
        ]

        return line


class KHybridGenerator(LCMDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

    def _gen_row_data(self) -> list:
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T", "Q"]
        self.header = cost_header + workload_header + system_header + decision

    def _sample_q(self, max_size_ratio: int) -> int:
        return np.random.randint(
            low=self.size_ratio_min - 1,
            high=max_size_ratio,
        )

    def _sample_design(self, system: System) -> LSMDesign:
        design = super()._sample_design(system)
        h = design.h
        T = design.T
        Q = self._sample_q(int(T))
        design = LSMDesign(h=h, T=T, policy=Policy.QFixed, Q=Q)

        return design

    def _gen_row_data(self) -> list:
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


class YZCostGenerator(LCMDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cost_header = self._gen_cost_header()
        workload_header = self._gen_workload_header()
        system_header = self._gen_system_header()
        decision = ["h", "T", "Y", "Z"]
        self.header = cost_header + workload_header + system_header + decision

    def _sample_capacity(self, max_size_ratio: int) -> int:
        return np.random.randint(
            low=self.size_ratio_min - 1,
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

    def _gen_row_data(self) -> list:
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
