import random
from typing import List, Optional, override
from itertools import combinations_with_replacement

import numpy as np

from axe.lsm.types import LSMDesign, System, Policy, LSMBounds, Workload
from axe.lsm.cost import EndureCost


class LSMDataGenerator:
    # Memory budget to prevent bits_per_elem from hitting too close to max, and
    # always ensuring write_buffer > 0
    MEM_EPSILON = 0.1

    def __init__(
        self,
        bounds: LSMBounds,
        precision: int = 3,
    ) -> None:
        self.precision = precision
        self.bounds = bounds
        self.max_levels = bounds.max_considered_levels
        self.cf = EndureCost(max_levels=bounds.max_considered_levels)

    def _sample_size_ratio(self) -> int:
        low, high = self.bounds.size_ratio_range
        return np.random.randint(low=low, high=high)

    def _sample_bloom_filter_bits(self, max: Optional[float] = None) -> float:
        if max is None:
            max = self.bounds.bits_per_elem_range[1]
        min = self.bounds.bits_per_elem_range[0]
        sample = (max - min) * np.random.rand() + min
        return np.around(sample, self.precision)

    # TODO: Will want to configure environment to simulate larger ranges over
    # potential system values
    def _sample_entry_per_page(self, entry_size: int = 8192) -> int:
        # Potential page sizes are 4KB, 8KB, 16KB
        KB_TO_BITS = 8 * 1024
        page_sizes = np.array(self.bounds.page_sizes)
        entries_per_page = (page_sizes * KB_TO_BITS) / entry_size
        return np.random.choice(entries_per_page)

    def _sample_selectivity(self) -> float:
        low, high = self.bounds.selectivity_range
        return (high - low) * np.random.rand() + low

    def _sample_entry_size(self) -> int:
        return np.random.choice(self.bounds.entry_sizes)

    def _sample_memory_budget(self) -> float:
        low, high = self.bounds.memory_budget_range
        return (high - low) * np.random.rand() + low

    def _sample_total_elements(self) -> int:
        low, high = self.bounds.elements_range
        return np.random.randint(low=low, high=high)

    def sample_system(self) -> System:
        E = self._sample_entry_size()
        B = self._sample_entry_per_page(entry_size=E)
        s = self._sample_selectivity()
        H = self._sample_memory_budget()
        N = self._sample_total_elements()
        system = System(E, s, B, N, H)

        return system

    def sample_design(
        self,
        system: System,
    ) -> LSMDesign:
        h = self._sample_bloom_filter_bits(max=(system.H - self.MEM_EPSILON))
        T = self._sample_size_ratio()
        lsm = LSMDesign(h, T)

        return lsm

    def sample_workload(self, dimensions: int) -> Workload:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629
        workload = np.around(np.random.rand(dimensions - 1), self.precision)
        workload = np.concatenate((workload, np.array([0, 1])))
        workload = np.sort(workload)

        workload = [b - a for a, b in zip(workload, workload[1:])]
        return Workload(*workload)


class TieringGenerator(LSMDataGenerator):
    def __init__(
        self,
        bounds: LSMBounds,
        policies: List[Policy] = [Policy.Tiering, Policy.Leveling],
        **kwargs,
    ):
        super().__init__(bounds, **kwargs)
        self.policies = policies

    @override
    def sample_design(
        self,
        system: System,
    ) -> LSMDesign:
        h = self._sample_bloom_filter_bits(max=(system.H - self.MEM_EPSILON))
        T = self._sample_size_ratio()
        lsm = LSMDesign(h, T, policy=Policy.Tiering)

        return lsm


class LevelingGenerator(LSMDataGenerator):
    def __init__(
        self,
        bounds: LSMBounds,
        policies: List[Policy] = [Policy.Tiering, Policy.Leveling],
        **kwargs,
    ):
        super().__init__(bounds, **kwargs)
        self.policies = policies

    @override
    def sample_design(
        self,
        system: System,
    ) -> LSMDesign:
        h = self._sample_bloom_filter_bits(max=(system.H - self.MEM_EPSILON))
        T = self._sample_size_ratio()
        lsm = LSMDesign(h, T, policy=Policy.Leveling)

        return lsm


class ClassicGenerator(LSMDataGenerator):
    def __init__(
        self,
        bounds: LSMBounds,
        **kwargs,
    ):
        super().__init__(bounds, **kwargs)

    @override
    def sample_design(
        self,
        system: System,
    ) -> LSMDesign:
        h = self._sample_bloom_filter_bits(max=(system.H - self.MEM_EPSILON))
        T = self._sample_size_ratio()
        policy = random.choice((Policy.Tiering, Policy.Leveling))
        lsm = LSMDesign(h=h, T=T, policy=policy)

        return lsm


class KHybridGenerator(LSMDataGenerator):
    def __init__(self, bounds: LSMBounds, **kwargs):
        super().__init__(bounds, **kwargs)

    def _gen_k_levels(self, levels: int, max_size_ratio: int) -> list:
        arr = combinations_with_replacement(range(max_size_ratio, 0, -1), levels)

        return list(arr)

    @override
    def sample_design(self, system: System) -> LSMDesign:
        design = super().sample_design(system)
        h = design.h
        T = design.T
        levels = int(self.cf.L(design, system, ceil=True))
        k = np.random.randint(low=1, high=int(T), size=(levels))
        remaining = np.ones(self.max_levels - len(k))
        k = np.concatenate([k, remaining])
        design = LSMDesign(h=h, T=T, policy=Policy.KHybrid, K=k.tolist())

        return design


class QCostGenerator(LSMDataGenerator):
    def __init__(self, bounds: LSMBounds, **kwargs):
        super().__init__(bounds, **kwargs)

    def _sample_q(self, max_size_ratio: int) -> int:
        return np.random.randint(
            low=self.bounds.size_ratio_range[0] - 1,
            high=max_size_ratio,
        )

    @override
    def sample_design(self, system: System) -> LSMDesign:
        design = super().sample_design(system)
        h = design.h
        T = design.T
        Q = self._sample_q(int(T))
        design = LSMDesign(h=h, T=T, policy=Policy.QFixed, Q=Q)

        return design


class YZCostGenerator(LSMDataGenerator):
    def __init__(self, bounds: LSMBounds, **kwargs):
        super().__init__(bounds, **kwargs)

    def _sample_capacity(self, max_size_ratio: int) -> int:
        return np.random.randint(
            low=self.bounds.size_ratio_range[0] - 1,
            high=max_size_ratio,
        )

    @override
    def sample_design(self, system: System) -> LSMDesign:
        design = super().sample_design(system)
        h = design.h
        T = design.T
        Y = self._sample_capacity(int(T))
        Z = self._sample_capacity(int(T))
        design = LSMDesign(h=h, T=T, policy=Policy.YZHybrid, Y=Y, Z=Z)

        return design
