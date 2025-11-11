#!/usr/bin/env python
import logging

import click
import toml

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
def axe(ctx: click.Context, config_path: str):
    """A CLI for the AXE project."""
    with open(config_path) as f:
        config = toml.load(f)

    logging.basicConfig(
        format=config["log"]["format"], datefmt=config["log"]["datefmt"]
    )
    log: logging.Logger = logging.getLogger(config["app"]["name"])
    log.setLevel(getattr(logging, config["log"]["level"]))
    log_level = logging.getLevelName(log.getEffectiveLevel())
    log.debug(f"Log level: {log_level}")

    ctx.obj = config


axe.add_command(create_lcm_data)
axe.add_command(create_ltuner_data)
axe.add_command(run_experiments)
axe.add_command(train_lcm)
axe.add_command(train_ltuner)
axe.add_command(train_robust_ltuner)

if __name__ == "__main__":
    axe()

