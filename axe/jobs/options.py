import click


def common_options(func):
    func = click.option(
        "--policy",
        default=None,
        type=click.Choice(
            ["Tiering", "Leveling", "Classic", "QHybrid", "Fluid", "Kapacity"],
            case_sensitive=False,
        ),
        help="Design space policy to consider for LSM tree",
    )(func)
