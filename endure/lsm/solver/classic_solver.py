from typing import Any, Optional, Callable, Tuple, List

import numpy as np
import scipy.optimize as SciOpt

from endure.lsm.cost import EndureCost
from endure.lsm.types import LSMDesign, Policy, System
from .util import kl_div_con
from .util import get_bounds

H_DEFAULT = 5
T_DEFAULT = 10
LAMBDA_DEFAULT = 1
ETA_DEFAULT = 1


class ClassicSolver:
    def __init__(
        self,
        config: dict[str, Any],
        policies: Optional[List[Policy]] = None
    ):
        self.config = config
        self.cf = EndureCost(config)
        if policies is None:
            policies = [Policy.Tiering, Policy.Leveling]
        self.policies = policies

    def robust_objective(
        self,
        x: np.ndarray,
        policy: Policy,
        system: System,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, lamb, eta = x
        design = LSMDesign(h=h, T=T, policy=policy)
        query_cost = 0
        query_cost += z0 * kl_div_con((self.cf.Z0(design, system) - eta) / lamb)
        query_cost += z1 * kl_div_con((self.cf.Z1(design, system) - eta) / lamb)
        query_cost += q * kl_div_con((self.cf.Q(design, system) - eta) / lamb)
        query_cost += w * kl_div_con((self.cf.W(design, system) - eta) / lamb)
        cost = eta + (rho * lamb) + (lamb * query_cost)
        return cost

    def nominal_objective(
        self,
        x: np.ndarray,
        policy: Policy,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ):
        h, T = x
        design = LSMDesign(h=h, T=T, policy=policy)
        cost = self.cf.calc_cost(design, system, z0, z1, q, w)

        return cost

    def get_robust_design(
        self,
        system: System,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: np.ndarray = np.array(
            [H_DEFAULT, T_DEFAULT, LAMBDA_DEFAULT, ETA_DEFAULT]
        ),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        design = None
        solution = None

        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                config=self.config,
                system=system,
                robust=True,
            ),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        min_sol = np.inf
        assert len(self.policies) > 0
        for policy in self.policies:
            sol = SciOpt.minimize(
                fun=lambda x: self.robust_objective(
                    x, policy, system, rho, z0, z1, q, w
                ),
                x0=init_args,
                callback=callback_fn,
                **default_kwargs
            )
            if sol.fun < min_sol or (design is None and solution is None):
                min_sol = sol.fun
                design = LSMDesign(sol.x[0], sol.x[1], policy=policy)
                solution = sol
        assert design is not None
        assert solution is not None

        return design, solution

    def get_nominal_design(
        self,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: np.ndarray = np.array([H_DEFAULT, T_DEFAULT]),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                config=self.config,
                system=system,
                robust=False,
            ),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        design, solution = None, None
        min_sol = np.inf
        for policy in self.policies:
            sol = SciOpt.minimize(
                fun=lambda x: self.nominal_objective(x, policy, system, z0, z1, q, w),
                x0=init_args,
                callback=callback_fn,
                **default_kwargs
            )
            if sol.fun < min_sol or (design is None and solution is None):
                min_sol = sol.fun
                design = LSMDesign(sol.x[0], sol.x[1], policy=policy)
                solution = sol
        assert design is not None
        assert solution is not None

        return design, solution
