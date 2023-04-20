#!/usr/bin/env python
import csv
import logging
import multiprocessing as mp
import os

from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from endure.data.io import Reader
import endure.lcm.data.generator as Generators


class LCMDataGenJob:
    def __init__(self, config):
        self.log = logging.getLogger(config["log"]["name"])
        self.log.info("Running Data Generator Job")
        self.config = config
        self.setting = config["job"]["LCMDataGen"]
        self.output_dir = os.path.join(
            self.config["io"]["data_dir"], self.setting["dir"]
        )

    def _choose_generator(self) -> Generators.LCMDataGenerator:
        choice = self.setting["generator"]
        generators = {
            "TierCost": Generators.TierGenerator,
            "LevelCost": Generators.LevelGenerator,
            "QCost": Generators.QCostGenerator,
            "KHybridCost": Generators.KHybridGenerator,
        }
        generator = generators.get(choice, None)
        if generator is None:
            self.log.error("Invalid generator choice. " "Defaulting to KHybridCost")
            return Generators.KHybridGenerator(self.config)

        generator = generator(self.config)
        return generator

    def generate_csv_file(self, generator, idx: int, pos: int) -> int:
        fname_prefix = self.setting["file_prefix"]
        fname = f"{fname_prefix}-{idx:04}.csv"
        fpath = os.path.join(self.output_dir, fname)
        if os.path.exists(fpath) and (not self.setting["overwrite_if_exists"]):
            self.log.info(f"{fpath} exists, exiting.")
            return -1

        samples = range(int(self.setting["samples"]))
        header = generator.generate_header()
        with open(fpath, "w") as fid:
            writer = csv.writer(fid)
            writer.writerow(header)
            for _ in tqdm(
                samples,
                desc=fname,
                position=pos,
                ncols=80,
                disable=self.config["log"]["disable_tqdm"],
            ):
                row = generator.generate_row()
                writer.writerow(row)

        return idx

    def generate_parquet_file(
        self, generator: Generators.LCMDataGenerator, idx: int, pos: int
    ) -> int:
        fname_prefix = self.setting["file_prefix"]
        fname = f"{fname_prefix}-{idx:04}.parquet"
        fpath = os.path.join(self.output_dir, fname)
        if os.path.exists(fpath) and (not self.setting["overwrite_if_exists"]):
            self.log.info(f"{fpath} exists, exiting.")
            return -1

        samples = range(int(self.setting["samples"]))
        table = []
        for _ in tqdm(
            samples,
            desc=fname,
            position=pos,
            ncols=80,
            disable=self.config["log"]["disable_tqdm"],
        ):
            table.append(generator.generate_row_parquet())
        table = pa.Table.from_pylist(table)
        pq.write_table(table, fpath)

        return idx

    def generate_file_single_thread(self) -> None:
        generator = self._choose_generator()

        if self.setting["format"] == "parquet":
            file_gen = self.generate_parquet_file
        else:  # format == 'csv'
            file_gen = self.generate_csv_file

        for idx in range(self.setting["num_files"]):
            file_gen(generator, idx, 0)

    def generate_file(self, idx: int) -> int:
        pos = 0
        if len(mp.current_process()._identity) > 0:
            pos = mp.current_process()._identity[0] - 1
        generator = self._choose_generator()

        if self.setting["format"] == "parquet":
            self.generate_parquet_file(generator, idx, pos)
        else:  # format == 'csv'
            self.generate_csv_file(generator, idx, pos)

        return idx

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f"Writing all files to {self.output_dir}")

        inputs = list(range(0, self.setting["num_files"]))
        threads = self.setting["num_workers"]
        if threads == -1:
            threads = mp.cpu_count()
        self.log.info(f"{threads=}")

        if threads == 1:
            self.generate_file_single_thread()
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

    a = LCMDataGenJob(config)
    a.run()
