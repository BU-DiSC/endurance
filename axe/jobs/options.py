from pathlib import Path

import click


# Common options for all model subcommands
def common_options(func):
    """Decorator to add common options to model subcommands."""
    func = click.option(
        "--use_gpu",
        is_flag=True,
        help="use gpu if available",
    )(func)

    func = click.option(
        "--disable-tqdm",
        is_flag=True,
        default=False,
        help="disable terminal progress bars (tqdm)",
    )(func)

    func = click.option(
        "--seed",
        type=int,
        help="random seed",
    )(func)

    func = click.option(
        "--verbose",
        "-v",
        is_flag=True,
        default=False,
        help="enable verbose (DEBUG) logging",
    )(func)

    func = click.option(
        "--config",
        "-c",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to YAML configuration file",
    )(func)

    return func


def training_options(func):
    func = click.option(
        "--batch-size",
        type=int,
        help="number of samples to per batch",
    )(func)

    return func
