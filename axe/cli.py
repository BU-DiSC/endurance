#!/usr/bin/env python
import logging

import toml
import typer
from typing_extensions import Annotated

from .config import AxeConfig
from .jobs.create_lcm_data import create_lcm_data
from .jobs.create_ltuner_data import create_ltuner_data
from .jobs.run_experiments import run_experiments
from .jobs.train_lcm import train_lcm
from .jobs.train_ltuner import train_ltuner
from .jobs.train_robust_ltuner import train_robust_ltuner

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Annotated[
        str,
        typer.Option(
            "--config",
            help="Path to the configuration file.",
        ),
    ] = "axe.toml",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging.",
        ),
    ] = False,
):
    """A CLI for the AXE project."""

    with open(config_path) as f:
        data = toml.load(f)
    config = AxeConfig.model_validate(data)

    format = "[%(levelname)s][%(asctime)-15s][%(filename)s] %(message)s"
    datefmt = "%d-%m-%y:%H:%M:%S"
    logging_level = logging.DEBUG if verbose else config.log_level 
    logging.basicConfig(level=logging_level, format=format, datefmt=datefmt)

    ctx.obj = config


app.command()(create_lcm_data)
app.command()(create_ltuner_data)
app.command()(run_experiments)
app.command()(train_lcm)
app.command()(train_ltuner)
app.command()(train_robust_ltuner)

if __name__ == "__main__":
    app()

