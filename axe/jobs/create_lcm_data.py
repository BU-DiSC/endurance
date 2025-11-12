import logging
import multiprocessing as mp
import os

import click
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from axe.lcm.data.schema import LCMDataSchema
from axe.lsm.types import LSMBounds, Policy

logger = logging.getLogger(__name__)


class CreateLCMData:
    def __init__(
        self,
        config: dict,
        output_dir: str,
        num_samples: int = 1024,
        num_threads: int = 1,
        num_files: int = 1,
        overwrite_if_exists: bool = False,
    ) -> None:
        self.disable_tqdm: bool = config["app"]["disable_tqdm"]
        self.policy: Policy = getattr(Policy, config["lsm"]["policy"])
        self.bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
        self.seed: int = config["app"]["random_seed"]

        self.output_dir: str = output_dir
        self.num_samples: int = num_samples
        self.num_files: int = num_files
        self.num_threads: int = num_threads
        self.overwrite_if_exists: bool = overwrite_if_exists
        self.cfg = config

    def generate_parquet_file(self, schema: LCMDataSchema, idx: int, pos: int) -> int:
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
        schema = LCMDataSchema(self.policy, self.bounds, seed=(self.seed + idx))

        self.generate_parquet_file(schema, idx, pos)

        return idx

    def run(self) -> None:
        logger.info("[Job] Creating LCM Data")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Writing all files to {self.output_dir}")

        inputs = list(range(0, self.num_files))
        threads = self.num_threads
        if threads == -1:
            threads = mp.cpu_count()
        if threads > self.num_files:
            logger.info("Num workers > num files, scaling down")
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


@click.command(
    "create-lcm-data",
    help="Generate training data for the Learned Cost Model (LCM).",
)
@click.option("--output-dir", help="Directory to save the generated data.")
@click.option(
    "--num-samples", type=int, default=1024, help="Number of samples per file."
)
@click.option("--num-files", type=int, default=1, help="Number of files to generate.")
@click.option("--num-threads", type=int, default=1, help="Number of threads to use.")
@click.option(
    "--overwrite-if-exists/--no-overwrite-if-exists",
    is_flag=True,
    default=False,
    help="Overwrite existing files if they exists.",
)
@click.option("--lsm-policy", "policy", help="LSM policy to use.")
@click.pass_context
def create_lcm_data(
    ctx: click.Context,
    output_dir: str,
    num_samples: int,
    num_files: int,
    num_threads: int,
    overwrite_if_exists: bool,
    policy: str,
):
    """Generate training data for the Learned Cost Model (LCM)."""
    config = ctx.obj

    if policy is not None:
        config["lsm"]["policy"] = policy

    CreateLCMData(
        config, output_dir, num_samples, num_threads, num_files, overwrite_if_exists
    ).run()
