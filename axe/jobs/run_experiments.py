#!/usr/bin/env python
import logging
from typing import Optional

import typer
from typing_extensions import Annotated

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


def run_experiments(
    ctx: typer.Context,
    exp_list: Annotated[
        list[str],
        typer.Option(
            "--exp",
            help="Specify experiments to run. Can be used multiple times.",
        ),
    ],
):
    """Run experiments."""
    config = ctx.obj

    if exp_list:
        config["job"]["run_experiments"]["exp_list"] = exp_list

    RunExperiments(config).run()
