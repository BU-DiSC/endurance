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
import endure.lcm.data.generator as Generators


class LCMDataGenJob:
    def __init__(self, config):
        self.log = logging.getLogger(config["log"]["name"])
        self.log.info("Running Data Generator Job")
        self.config = config
        self.jconfig = config["job"]["LCMDataGen"]

    def create_bounds(self) -> LSMBounds:
        return LSMBounds(**self.config["lsm"]["bounds"])

    def _choose_generator(self) -> Generators.LCMDataGenerator:
        design_enum = getattr(Policy, self.config["lsm"]["design"])
        bounds = self.create_bounds()
        self.log.info(f"Generator: {design_enum.name}")
        self.log.info(f"{bounds=}")
        generators = {
            Policy.Tiering: Generators.ClassicGenerator,
            Policy.Leveling: Generators.ClassicGenerator,
            Policy.Classic: Generators.ClassicGenerator,
            Policy.QFixed: Generators.QCostGenerator,
            Policy.KHybrid: Generators.KHybridGenerator,
        }
        generator = generators.get(design_enum, None)
        if generator is None:
            raise TypeError("Invalid generator choice")

        gen_kwargs: dict[str, Any] = {
            "bounds": bounds,
        }
        if design_enum in [Policy.Tiering, Policy.Leveling]:
            gen_kwargs["policies"] = [design_enum]
        elif design_enum == Policy.Classic:
            gen_kwargs["policies"] = [Policy.Tiering, Policy.Leveling]
        generator = generator(**gen_kwargs)

        return generator

    def generate_parquet_file(
        self, generator: Generators.LCMDataGenerator, idx: int, pos: int
    ) -> int:
        fname_prefix = self.jconfig["file_prefix"]
        fname = f"{fname_prefix}_{idx:04}.parquet"
        fpath = os.path.join(
            self.config["io"]["data_dir"], self.jconfig["dir"], fname
        )
        if os.path.exists(fpath) and (not self.jconfig["overwrite_if_exists"]):
            self.log.info(f"{fpath} exists, exiting.")
            return -1

        samples = range(int(self.jconfig["samples"]))
        table = []
        for _ in tqdm(
            samples,
            desc=fname,
            position=pos,
            ncols=80,
            disable=self.config["log"]["disable_tqdm"],
        ):
            table.append(generator.generate_row())
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
        data_dir = os.path.join(self.config["io"]["data_dir"], self.jconfig["dir"])
        os.makedirs(data_dir, exist_ok=True)
        self.log.info(f"Writing all files to {data_dir}")

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

    job = LCMDataGenJob(config)
    job.run()
