import logging
import numpy as np


class LTuneGenerator:
    def __init__(
        self,
        config: dict[str, ...],
        format: str = 'parquet',
        precision: int = 3
    ):
        self.log = logging.getLogger(config['log']['name'])
        self._config = config
        self._header = ['z0', 'z1', 'q', 'w']
        self.format = format
        self.precision = precision

    def _sample_workload(self, dimensions: int) -> list:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629
        workload = np.around(np.random.rand(dimensions - 1), self.precision)
        workload = np.concatenate([workload, [0, 1]])
        workload = np.sort(workload)

        return [b - a for a, b in zip(workload, workload[1:])]

    def generate_row_csv(self) -> list:
        z0, z1, q, w = self._sample_workload(4)

        line = [z0, z1, q, w]
        return line

    def generate_row_parquet(self) -> list:
        header = self.generate_header()
        row = self.generate_row_csv()
        line = {}
        for key, val in zip(header, row):
            line[key] = val

        return line

    def generate_header(self) -> list:
        return self._header

    def generate_row(self) -> list:
        if self.format == 'parquet':
            row = self.generate_row_parquet()
        else:  # format == 'csv'
            row = self.generate_row_csv()

        return row

    def generate_workloads(self, num_samples: int, seed: int = 0) -> list:
        workloads = []
        np.random.seed(0)
        for _ in range(num_samples):
            workloads.append(self._sample_workload(4))

        return workloads
