from typing import Any, Optional, Callable, Tuple, List

import numpy as np
import scipy.optimize as SciOpt

from endure.lsm.cost import EndureCost
from endure.lsm.types import LSMDesign, Policy, System
from .util import kl_div_con, get_bounds
from .util import H_DEFAULT, T_DEFAULT, LAMBDA_DEFAULT, ETA_DEFAULT, Q_DEFAULT

class QLSMSolver:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.cf = EndureCost(config)

    def robust_objective(
        self,
        x: np.ndarray,
        system: System,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, t, q, lamb, eta = x
        design = LSMDesign(h=h, T=t, Q=q, policy=Policy.QFixed)
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
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, t, q = x
        design = LSMDesign(h=h, T=t, Q=q, policy=Policy.QFixed)
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
        raise NotImplemented

    def get_nominal_design(
        self,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: np.ndarray = np.array([H_DEFAULT, T_DEFAULT, Q_DEFAULT]),
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

        solution = SciOpt.minimize(
            fun=lambda x: self.nominal_objective(x, system, z0, z1, q, w),
            x0=init_args,
            callback=callback_fn,
            **default_kwargs
        )
        design = LSMDesign(
            h=solution.x[0],
            T=solution.x[1],
            Q=solution.x[2],
            policy=Policy.QFixed)

        return design, solution
