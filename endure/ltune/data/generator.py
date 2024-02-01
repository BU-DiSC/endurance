from typing import List, Tuple, Union

import numpy as np

from endure.lsm.types import System
from endure.ltune.data.input_features import kSYSTEM_HEADER, kWORKLOAD_HEADER


class LTuneDataGenerator:
    def __init__(
        self,
        page_sizes: List[int] = [4, 8, 16],
        entry_sizes: List[int] = [1024, 2048, 4096, 8192],
        memory_budget_range: Tuple[float, float] = (5.0, 20.0),
        selectivity_range: Tuple[float, float] = (1e-7, 1e-9),
        elements_range: Tuple[int, int] = (100000000, 1000000000),
        precision: int = 3,
    ) -> None:
        self.entry_sizes = entry_sizes
        self.memory_budget_range = memory_budget_range
        self.page_sizes = page_sizes
        self.selectivity_range = selectivity_range
        self.elements_range = elements_range
        self.precision = precision

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

    def _gen_system_header(self) -> list:
        return kSYSTEM_HEADER

    def _gen_workload_header(self) -> list:
        return kWORKLOAD_HEADER

    def generate_header(self) -> list:
        return self._gen_workload_header() + self._gen_system_header()

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)
        system: System = self._sample_system()

        line = [
            z0,
            z1,
            q,
            w,
            system.B,
            system.s,
            system.E,
            system.H,
            system.N,
        ]

        return line

    def generate_row(self) -> dict[str, Union[int, float]]:
        header = self.generate_header()
        row = self.generate_row_csv()
        line = {}
        for key, val in zip(header, row):
            line[key] = val

        return line
