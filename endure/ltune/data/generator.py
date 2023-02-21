import logging
import numpy as np


class LTunerGenerator:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self._header = ['z0', 'z1', 'q', 'w']
        self.log = logging.getLogger('endure')

    def _sample_workload(self, dimensions: int) -> list:
        # See stackoverflow thread for why the simple solution is not uniform
        # https://stackoverflow.com/questions/8064629
        workload = list(np.random.rand(dimensions - 1)) + [0, 1]
        workload.sort()

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
        if self._config['data']['gen']['format'] == 'parquet':
            row = self.generate_row_parquet()
        else:  # format == 'csv'
            row = self.generate_row_csv()

        return row
