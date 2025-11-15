from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.optimize as SciOpt

from axe.config import LSMBounds
from axe.lsm.cost import Cost
from axe.lsm.types import LSMDesign, Policy, System, Workload

from .util import get_bounds, kl_div_con

H_DEFAULT = 3
T_DEFAULT = 5
LAMBDA_DEFAULT = 10
ETA_DEFAULT = 10


class ClassicSolver:
    def __init__(self, bounds: LSMBounds, policies: Optional[List[Policy]] = None):
        self.bounds = bounds
        self.costfunc = Cost(bounds.max_considered_levels)
        if policies is None:
            policies = [Policy.Tiering, Policy.Leveling]
        self.policies = policies

    def robust_objective(
        self,
        x: np.ndarray,
        policy: Policy,
        system: System,
        workload: Workload,
        rho: float,
    ) -> float:
        (
            eta,
            lamb,
            h,
            T,
        ) = x
        design = LSMDesign(bits_per_elem=h, size_ratio=T, policy=policy, kapacity=())
        query_cost = 0
        query_cost += workload.z0 * kl_div_con(
            (self.costfunc.Z0(design, system) - eta) / lamb
        )
        query_cost += workload.z1 * kl_div_con(
            (self.costfunc.Z1(design, system) - eta) / lamb
        )
        query_cost += workload.q * kl_div_con(
            (self.costfunc.Q(design, system) - eta) / lamb
        )
        query_cost += workload.w * kl_div_con(
            (self.costfunc.W(design, system) - eta) / lamb
        )
        cost = eta + (rho * lamb) + (lamb * query_cost)
        return cost

    def nominal_objective(
        self,
        x: np.ndarray,
        policy: Policy,
        system: System,
        workload: Workload,
    ):
        h, T = x
        design = LSMDesign(bits_per_elem=h, size_ratio=T, policy=policy, kapacity=())
        cost = self.costfunc.calc_cost(design, system, workload)

        return cost

    def get_robust_design(
        self,
        system: System,
        workload: Workload,
        rho: float,
        init_args: np.ndarray = np.array(
            [ETA_DEFAULT, LAMBDA_DEFAULT, H_DEFAULT, T_DEFAULT]
        ),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        design = None
        solution = None

        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                system=system,
                robust=True,
            ),
            "options": {"ftol": 1e-12, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        min_sol = np.inf
        assert len(self.policies) > 0
        for policy in self.policies:
            sol = SciOpt.minimize(
                fun=lambda x: self.robust_objective(x, policy, system, workload, rho),
                x0=init_args,
                callback=callback_fn,
                **default_kwargs,
            )
            if sol.fun < min_sol or (design is None and solution is None):
                min_sol = sol.fun
                design = LSMDesign(
                    bits_per_elem=sol.x[2],
                    size_ratio=sol.x[3],
                    policy=policy,
                    kapacity=(),
                )
                solution = sol
        assert design is not None
        assert solution is not None

        return design, solution

    def get_nominal_design(
        self,
        system: System,
        workload: Workload,
        init_args: np.ndarray = np.array([H_DEFAULT, T_DEFAULT]),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                system=system,
                robust=False,
            ),
            "options": {"ftol": 1e-12, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        design, solution = None, None
        min_sol = np.inf
        for policy in self.policies:
            sol = SciOpt.minimize(
                fun=lambda x: self.nominal_objective(x, policy, system, workload),
                x0=init_args,
                callback=callback_fn,
                **default_kwargs,
            )
            if sol.fun < min_sol or (design is None and solution is None):
                min_sol = sol.fun
                design = LSMDesign(
                    bits_per_elem=sol.x[0],
                    size_ratio=sol.x[1],
                    policy=policy,
                    kapacity=(),
                )
                solution = sol
        assert design is not None
        assert solution is not None

        return design, solution
