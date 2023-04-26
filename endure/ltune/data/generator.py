from copy import deepcopy
from typing import Any, Union
import logging

import numpy as np


class LTuneGenerator:
    def __init__(
        self, config: dict[str, Any], format: str = "parquet", precision: int = 3
    ):
        self.log = logging.getLogger(config["log"]["name"])
        self.log.debug("Init LTuneGenerator")
        self._config = config
        self._header = self._gen_workload_header() + self._gen_system_header()
        self.format = format
        self.precision = precision

    def _sample_workload(self, dimensions: int) -> list:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629
        workload = np.around(np.random.rand(dimensions - 1), self.precision)
        workload = np.concatenate([workload, [0, 1]])
        workload = np.sort(workload)

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

    def _gen_system_header(self) -> list:
        return ["B", "s", "E", "H", "N"]

    def _gen_workload_header(self) -> list:
        return ["z0", "z1", "q", "w"]

    def generate_header(self) -> list:
        return self._gen_workload_header() + self._gen_system_header()

    def generate_row_csv(self) -> list[float]:
        z0, z1, q, w = self._sample_workload(4)
        B, s, E, H, N = self._sample_system()

        config = deepcopy(self._config)
        config["lsm"]["system"]["B"] = B
        config["lsm"]["system"]["s"] = s
        config["lsm"]["system"]["E"] = E
        config["lsm"]["system"]["H"] = H
        config["lsm"]["system"]["N"] = N

        return [z0, z1, q, w, B, s, E, H, N]

    def generate_row_parquet(self) -> dict[str, Union[int, float]]:
        header = self.generate_header()
        row = self.generate_row_csv()
        line = {}
        for key, val in zip(header, row):
            line[key] = val

        return line

    def generate_row(self) -> Union[list, dict[str, Union[int, float]]]:
        if self.format == "parquet":
            row = self.generate_row_parquet()
        else:  # format == 'csv'
            row = self.generate_row_csv()

        return row
