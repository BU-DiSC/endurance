#!/usr/bin/env python
import logging

import click

from ..experiments.evaluate_ltuner import ExpLTunerEvaluate
from ..experiments.mlos_exp_runs import ExperimentMLOS
from ..experiments.evaluate_lcm import ExpLCMEvaluate


class RunExperiments:
    def __init__(self, cfg: dict) -> None:
        self.log: logging.Logger = logging.getLogger(cfg["app"]["name"])

        jcfg = cfg["job"]["run_experiments"]
        self.exp_list = jcfg["exp_list"]
        self.cfg = cfg

    def run(self) -> None:
        experiments = {
            "ExperimentMLOS": ExperimentMLOS,
            "ExpLCMEvaluate": ExpLCMEvaluate,
            "ExpLTunerEvaluate": ExpLTunerEvaluate
        }
        self.log.info(f"Jobs to run: {self.exp_list}")
        for exp_name in self.exp_list:
            experiment = experiments.get(exp_name, None)
            if experiment is None:
                self.log.warning(f"No job associated with {exp_name}")
                continue
            exp = experiment(self.cfg)
            _ = exp.run()

        self.log.info("All experiments finished, exiting")


@click.command("run-experiments", help="Run experiments.")
@click.option(
    "--exp",
    "exp_list",
    multiple=True,
    help="Specify experiments to run. Can be used multiple times.",
)
@click.pass_context
def run_experiments(ctx: click.Context, exp_list: tuple[str, ...]):
    """Run experiments."""
    config = ctx.obj

    if exp_list:
        config["job"]["run_experiments"]["exp_list"] = list(exp_list)

    RunExperiments(config).run()
