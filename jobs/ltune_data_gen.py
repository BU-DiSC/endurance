#!/usr/bin/env python
import logging
import multiprocessing as mp
import os

from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from endure.data.io import Reader
from endure.lsm.types import LSMBounds
from endure.ltune.data.generator import LTuneDataGenerator


class LTuneDataGenJob:
    def __init__(self, config):
        self.log = logging.getLogger(config["log"]["name"])
        self.log.info("Starting LTuneDataGenJob")
        self.output_dir = os.path.join(
            config["io"]["data_dir"],
            config["job"]["LTuneDataGen"]["dir"],
        )
        self.bounds = LSMBounds(**config["lsm"]["bounds"])

        self.config = config
        self.jconfig = config["job"]["LTuneDataGen"]

    def _choose_generator(self):
        return LTuneDataGenerator(self.bounds)

    def generate_parquet_file(
        self,
        generator: LTuneDataGenerator,
        idx: int,
        pos: int,
    ) -> int:
        fname_prefix = self.jconfig["file_prefix"]
        fname = f"{fname_prefix}-{idx:04}.parquet"
        fpath = os.path.join(self.output_dir, fname)

        if os.path.exists(fpath) and not self.jconfig["overwrite_if_exists"]:
            self.log.info(f"{fpath} exists, exiting.")
            return -1

        samples = range(self.jconfig["samples"])
        pbar = tqdm(
            samples,
            desc=fname,
            position=pos,
            ncols=80,
            disable=self.config["log"]["disable_tqdm"],
        )
        table = [generator.generate_row() for _ in pbar]
        table = pa.Table.from_pylist(table)
        pq.write_table(table, fpath)

        return idx

    def generate_file(self, idx: int, single_threaded: bool = False) -> int:
        pos = 0
        if len(mp.current_process()._identity) > 0 and not single_threaded:
            pos = mp.current_process()._identity[0] - 1
        generator = self._choose_generator()
        self.generate_parquet_file(generator, idx, pos)

        return idx

    def run(self) -> None:
        self.log.info("Creating workload data")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f"Writing all files to {self.output_dir}")

        inputs = list(range(0, self.jconfig["num_files"]))
        threads = self.jconfig["num_workers"]
        if threads == -1:
            threads = mp.cpu_count()
        if threads > self.jconfig["num_files"]:
            self.log.info("Number of threads > num files, scaling down")
            threads = self.jconfig["num_files"]
        self.log.info(f"{threads=}")

        if threads == 1:
            self.generate_file(0, single_threaded=True)
        else:
            with mp.Pool(
                threads, initializer=tqdm.set_lock, initargs=(mp.RLock(),)
            ) as p:
                p.map(self.generate_file, inputs)

        return


if __name__ == "__main__":
    config = Reader.read_config("endure.toml")

    logging.basicConfig(
        format=config["log"]["format"], datefmt=config["log"]["datefmt"]
    )
    log = logging.getLogger(config["log"]["name"])
    log.setLevel(config["log"]["level"])

    job = LTuneDataGenJob(config)
    job.run()
