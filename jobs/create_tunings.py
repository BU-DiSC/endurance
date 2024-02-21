#!/usr/bin/env python
from typing import List
import logging
import numpy as np
import pandas as pd

from endure.lcm.data.generator import LCMDataGenerator
import endure.lsm.solver as Solvers
from endure.lsm.types import Policy


class CreateTuningsJob:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(self.config["log"]["name"])
        self.log.info("Running Create Tunings Job")

    def _generate_samples(self, samples=10000) -> List:
        dg = LCMDataGenerator(self.config)
        return [dg._sample_workload(4) for _ in range(samples)]

    def _generate_rhos(self, start=0, stop=4, step=0.25) -> np.ndarray:
        return np.arange(start, stop, step)

    def _generate_solver(self):
        choice = self.config["tunings"]["cost_model"]
        choices = {
            "KCost": Solvers.KLSMSolver(self.config),
            "QCost": Solvers.QLSMSolver(self.config),
            "YZCost": Solvers.YZLSMSolver(self.config),
            "TierCost": Solvers.ClassicSolver(self.config, policies=[Policy.Tiering]),
            "LevelCost": Solvers.ClassicSolver(self.config, policies=[Policy.Leveling]),
            "TierLevelCost": Solvers.ClassicSolver(self.config)
        }
        solver = choices.get(choice, None)
        if solver is None:
            self.log.error("Invalid cost model choice. Defaulting to ClassicSolver")
            return Solvers.ClassicSolver(self.config)

        return solver

    def run(self) -> pd.DataFrame:
        df = pd.DataFrame()

        return df
