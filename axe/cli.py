#!/usr/bin/env python
import logging

import click
import toml

from .config import AxeConfig
from .jobs.create_lcm_data import create_lcm_data
from .jobs.create_ltuner_data import create_ltuner_data
from .jobs.run_experiments import run_experiments
from .jobs.train_lcm import train_lcm
from .jobs.train_ltuner import train_ltuner
from .jobs.train_robust_ltuner import train_robust_ltuner


@click.group()
@click.option(
    "--config",
    "config_path",
    default="axe.toml",
    help="Path to the configuration file.",
    type=click.Path(exists=True),
)
@click.pass_context
def main(ctx: click.Context, config_path: str):
    """A CLI for the AXE project."""
    format = "[%(levelname)s][%(asctime)-15s][%(filename)s] %(message)s"
    datefmt = "%d-%m-%y:%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger: logging.Logger = logging.getLogger(__name__)

    with open(config_path) as f:
        data = toml.load(f)
    config = AxeConfig.model_validate(data)

    logger.setLevel(config.log_level)

    ctx.obj = config


main.add_command(create_lcm_data)
main.add_command(create_ltuner_data)
main.add_command(run_experiments)
main.add_command(train_lcm)
main.add_command(train_ltuner)
main.add_command(train_robust_ltuner)

if __name__ == "__main__":
    main()
