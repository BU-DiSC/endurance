from typing import Any, Optional, Callable, Tuple

import numpy as np
import scipy.optimize as SciOpt

from endure.lsm.cost import EndureCost
from endure.lsm.types import LSMDesign, Policy, System, LSMBounds
from .util import kl_div_con, get_bounds
from .util import H_DEFAULT, T_DEFAULT, LAMBDA_DEFAULT, ETA_DEFAULT, K_DEFAULT


class KLSMSolver:
    def __init__(self, bounds: LSMBounds):
        self.bounds = bounds
        self.cf = EndureCost(bounds.max_considered_levels)

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
        h, t = x[0:2]
        kaps = x[2:-2].tolist()
        lamb, eta = x[-2:]
        design = LSMDesign(h=h, T=t, K=kaps, policy=Policy.KHybrid)
        query_cost = 0
        query_cost += z0 * \
            kl_div_con((self.cf.Z0(design, system) - eta) / lamb)
        query_cost += z1 * \
            kl_div_con((self.cf.Z1(design, system) - eta) / lamb)
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
        h, t = x[0:2]
        kaps = x[2:].tolist()
        design = LSMDesign(h=h, T=t, K=kaps, policy=Policy.KHybrid)
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
        init_args: np.ndarray = np.array([H_DEFAULT, T_DEFAULT, K_DEFAULT]),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        max_levels = self.bounds.max_considered_levels

        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                policy=Policy.KHybrid,
                system=system,
                robust=False,
            ),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)
        kap_val = init_args[-1]
        init_args = np.concatenate(
            (init_args[0:2],
             np.array([kap_val for _ in range(max_levels)]))
        )

        solution = SciOpt.minimize(
            fun=lambda x: self.nominal_objective(x, system, z0, z1, q, w),
            x0=init_args,
            callback=callback_fn,
            **default_kwargs
        )
        design = LSMDesign(
            h=solution.x[0],
            T=solution.x[1],
            K=solution.x[2:],
            policy=Policy.KHybrid
        )

        return design, solution
