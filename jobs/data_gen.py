#!/usr/bin/env python
import logging
import multiprocessing as mp
import os
from typing import Any

from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from endure.data.io import Reader
from endure.lsm.types import Policy, LSMBounds
from endure.ltune.data.generator import LTuneDataGenerator
import endure.lcm.data.generator as Generators


class DataGenJob:
    def __init__(self, config):
        self.log = logging.getLogger(config["log"]["name"])
        self.log.info("Running Data Generator Job")
        self.output_dir = os.path.join(
            config["io"]["data_dir"],
            config["job"]["DataGen"]["dir"],
        )
        self.bounds = LSMBounds(**config["lsm"]["bounds"])
        self.design = getattr(Policy, config["lsm"]["design"])

        self.config = config
        self.jconfig = config["job"]["DataGen"]
        if self.jconfig["generator"] == "LTuner":
            self.log.info("Generating data for Learned Tuner")
        elif self.jconfig["generator"] == "LCM":
            self.log.info("Generating data for Learned Cost Models")
        else:
            self.log.critical("Invalid generator type")
            raise KeyError

    def create_bounds(self) -> LSMBounds:
        return LSMBounds(**self.config["lsm"]["bounds"])

    def _choose_generator(self) -> Generators.LCMDataGenerator | LTuneDataGenerator:
        if self.jconfig["generator"] == "LTuner":
            return LTuneDataGenerator(self.bounds)

        self.log.info(f"Generator: {self.design.name}")
        self.log.info(f"{self.bounds=}")
        generators = {
            Policy.Tiering: Generators.ClassicGenerator,
            Policy.Leveling: Generators.ClassicGenerator,
            Policy.Classic: Generators.ClassicGenerator,
            Policy.QFixed: Generators.QCostGenerator,
            Policy.KHybrid: Generators.KHybridGenerator,
        }
        generator = generators.get(self.design, None)
        if generator is None:
            raise TypeError("Invalid generator choice")

        gen_kwargs: dict[str, Any] = {"bounds": self.bounds}
        if self.design in [Policy.Tiering, Policy.Leveling]:
            gen_kwargs["policies"] = [self.design]
        elif self.design == Policy.Classic:
            gen_kwargs["policies"] = [Policy.Tiering, Policy.Leveling]
        generator = generator(**gen_kwargs)

        return generator

    def generate_parquet_file(
        self,
        generator: Generators.LCMDataGenerator | LTuneDataGenerator,
        idx: int,
        pos: int,
    ) -> int:
        fname_prefix = self.jconfig["file_prefix"]
        fname = f"{fname_prefix}_{idx:04}.parquet"
        fpath = os.path.join(self.output_dir, fname)

        if os.path.exists(fpath) and (not self.jconfig["overwrite_if_exists"]):
            self.log.info(f"{fpath} exists, exiting.")
            return -1

        pbar = tqdm(
            range(int(self.jconfig["samples"])),
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
            self.log.info("Num threads > num files, scaling down")
            threads = self.jconfig["num_files"]
        self.log.debug(f"Using {threads=}")

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

    job = DataGenJob(config)
    job.run()
