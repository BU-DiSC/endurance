#!/usr/bin/env python
import logging
import pandas as pd
import numpy as np
from typing import Optional
from copy import deepcopy
import lsm.solver as Solvers
from data.data_generators import DataGenerator

from tqdm import tqdm


class CreateTuningsJob:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(self.config['log']['name'])
        self.log.info('Running Create Tunings Job')

    def _generate_samples(self, samples=10000) -> list:
        dg = DataGenerator(self.config)
        return [dg._sample_workload(4) for _ in range(samples)]

    def _generate_rhos(self, start=0, stop=4, step=0.25) -> np.ndarray:
        return np.arange(start, stop, step)

    def _generate_solver(self) -> Solvers.EndureSolver:
        choice = self.config['tunings']['cost_model']
        choices = {
            'YZCost': Solvers.EndureYZSolver(self.config),
            'KCost': Solvers.EndureKSolver(self.config),
            'QCost': Solvers.EndureQSolver(self.config),
            'TierCost': Solvers.EndureTierSolver(self.config),
            'LevelCost': Solvers.EndureLevelSolver(self.config),
            'TierLevelCost': Solvers.EndureTierLevelSolver(self.config)
        }
        solver = choices.get(choice, None)
        if solver is None:
            self.log.error('Invalid cost model choice. '
                           'Defaulting to KCost')
            solver = choices.get('KCost')

        return solver

    def _generate_nominal_fields(
        self,
        solver: Solvers.EndureSolver,
        wl: dict
    ) -> dict:
        row = {}
        z0, z1, q, w = (wl['z0'], wl['z1'], wl['q'], wl['w'])
        row['z0'], row['z1'], row['q'], row['w'] = (z0, z1, q, w)
        nominal = solver.find_nominal_design(z0, z1, q, w)
        row['nominal_design'] = nominal.x
        row['nominal_cost'] = nominal.fun

        return row

    def _generate_robust_fields(
        self,
        rho: float,
        solver: Solvers.EndureSolver,
        wl: dict,
        nominal_design: Optional[dict] = None
    ) -> dict:
        row = {}
        z0, z1, q, w = (wl['z0'], wl['z1'], wl['q'], wl['w'])
        if nominal_design is None:
            robust = solver.find_robust_design(rho, z0, z1, q, w)
        else:
            robust = solver.find_robust_design(
                rho, z0, z1, q, w, nominal_design)
        row['robust_design'] = robust.x
        row['robust_cost'] = solver.nominal_objective(
            robust.x[0:-2],  # Manually remove eta and lambda
            z0, z1, q, w)

        return row

    def run(self, save_file=True) -> pd.DataFrame:
        df = []
        rhos = self._generate_rhos(
            start=self.config['tunings']['rho']['start'],
            stop=self.config['tunings']['rho']['stop'],
            step=self.config['tunings']['rho']['step'])
        solver = self._generate_solver()

        pbar = tqdm(self.config['workloads'], ncols=80)
        for idx, workload in enumerate(pbar):
            row = {}
            row['workload_idx'] = idx
            nom_row = self._generate_nominal_fields(solver, workload)
            row.update(nom_row)
            for rho in rhos:
                robust_row = self._generate_robust_fields(
                    rho, solver, workload, nom_row['nominal_design'])
                row.update(robust_row)
                df.append(deepcopy(row))

        df = pd.DataFrame(df)
        return df
