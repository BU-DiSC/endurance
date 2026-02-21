#!/usr/bin/env python
import logging
import multiprocessing as mp
import os

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from axe.config import AxeConfig, LSMBounds
from axe.lsm.types import Policy
from axe.ltuner.data.schema import LTunerDataSchema

logger = logging.getLogger(__name__)


class CreateLTunerData:
    def __init__(
        self,
        config: AxeConfig,
        output_dir: str,
        num_samples: int = 1024,
        num_threads: int = 1,
        num_files: int = 1,
        overwrite_if_exists: bool = False,
    ) -> None:
        self.disable_tqdm: bool = config.disable_tqdm
        self.policy: Policy = config.lsm.policy
        self.bounds: LSMBounds = config.lsm.bounds
        self.seed: int = config.seed

        self.output_dir: str = output_dir
        self.num_samples: int = num_samples
        self.num_files: int = num_files
        self.num_threads: int = num_threads
        self.overwrite_if_exists: bool = overwrite_if_exists
        self.config = config

    def generate_parquet_file(
        self, schema: LTunerDataSchema, idx: int, pos: int
    ) -> int:
        fname = f"data{idx:04}.parquet"
        fpath = os.path.join(self.output_dir, fname)

        if os.path.exists(fpath) and (not self.overwrite_if_exists):
            logger.debug(f"{fpath} exists, exiting.")
            return -1

        pbar = tqdm(
            range(self.num_samples),
            desc=fname,
            position=pos,
            ncols=80,
            disable=self.disable_tqdm,
        )
        table = [schema.sample_row_dict() for _ in pbar]
        table = pa.Table.from_pylist(table)
        pq.write_table(table, fpath)

        return idx

    def generate_file(self, idx: int, single_worker: bool = False) -> int:
        pos = 0
        if len(mp.current_process()._identity) > 0 and not single_worker:
            pos = mp.current_process()._identity[0] - 1
        schema = LTunerDataSchema(self.policy, self.bounds, seed=(self.seed + idx))

        self.generate_parquet_file(schema, idx, pos)

        return idx

    def run(self) -> None:
        logger.info("[Job] Creating LTuner Data")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Writing all files to {self.output_dir}")

        inputs = list(range(0, self.num_files))
        threads = self.num_threads
        if threads == -1:
            threads = mp.cpu_count()
        if threads > self.num_files:
            logger.debug("Num workers > num files, scaling down")
            threads = self.num_files
        logger.debug(f"Using {threads=}")

        if threads < 2:
            for idx in range(self.num_files):
                self.generate_file(idx, single_worker=True)
        else:
            with mp.Pool(
                threads, initializer=tqdm.set_lock, initargs=(mp.RLock(),)
            ) as p:
                p.map(self.generate_file, inputs)

        return


def create_ltuner_data(
    ctx: typer.Context,
    output_dir: Annotated[
        str, typer.Option("--output-dir", help="Directory to save the generated data.")
    ],
    num_samples: Annotated[
        int, typer.Option("--num-samples", help="Number of samples per file.")
    ],
    num_files: Annotated[
        int, typer.Option("--num-files", help="Number of files to generate.")
    ],
    num_workers: Annotated[
        int, typer.Option("--num-workers", help="Number of worker processes to use.")
    ],
    overwrite_if_exists: Annotated[
        bool, typer.Option("--overwrite-if-exists", help="Overwrite existing files.")
    ],
    policy: Annotated[str, typer.Option("--lsm-policy", help="LSM policy to use.")],
):
    """Generate training data for the Learned Tuner (LTuner)."""
    config = ctx.obj

    # Update config with CLI options if they are provided
    job_config = config["job"]["create_ltuner_data"]
    if output_dir is not None:
        job_config["output_dir"] = output_dir
    if num_samples is not None:
        job_config["num_samples"] = num_samples
    if num_files is not None:
        job_config["num_files"] = num_files
    if num_workers is not None:
        job_config["num_workers"] = num_workers
    if overwrite_if_exists:
        job_config["overwrite_if_exists"] = overwrite_if_exists
    if policy is not None:
        config["lsm"]["policy"] = policy

    CreateLTunerData(config, output_dir=output_dir).run()
