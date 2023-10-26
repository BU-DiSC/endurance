from typing import Any, Optional, Callable, Tuple

import numpy as np
import scipy.optimize as SciOpt

from endure.lsm.cost import EndureCost
from endure.lsm.types import LSMDesign, Policy, System

H_DEFAULT = 5
T_DEFAULT = 10
Z_DEFAULT = 3
Y_DEFAULT = 3
Q_DEFAULT = 3
K_DEFAULT = 3
LAMBDA_DEFAULT = 1
ETA_DEFAULT = 1


class ClassicSolver:
    def __init__(self, config: dict[str, Any], policies=None):
        self.config = config
        self.cf = EndureCost(config)
        if policies is None:
            policies = [Policy.Tiering, Policy.Leveling]
        self.policies = policies

    def kl_div_con(self, input):
        # if input > 709:  # Unfortuantely we overflow above this
        #     return np.finfo(np.float64).max
        return np.exp(input) - 1

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
        query_cost += z0 * self.kl_div_con((self.cf.Z0(design, system) - eta) / lamb)
        query_cost += z1 * self.kl_div_con((self.cf.Z1(design, system) - eta) / lamb)
        query_cost += q * self.kl_div_con((self.cf.Q(design, system) - eta) / lamb)
        query_cost += w * self.kl_div_con((self.cf.W(design, system) - eta) / lamb)
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

    def get_bounds(self, system: Optional[System] = None) -> SciOpt.Bounds:
        t_ub = self.config["lsm"]["size_ratio"]["max"]
        t_lb = self.config["lsm"]["size_ratio"]["min"]
        h_lb = self.config["lsm"]["bits_per_elem"]["min"]
        if system is None:
            h_ub = self.config["lsm"]["system"]["H"] - 0.1
        else:
            h_ub = system.H - 0.1

        lb = (h_lb, t_lb)
        ub = (h_ub, t_ub)
        keep_feasible = True

        # SciPy's typing with Bounds is not correct
        return SciOpt.Bounds(lb=lb, ub=ub, keep_feasible=keep_feasible)  # type: ignore

    def get_robust_bounds(self) -> SciOpt.Bounds:
        bounds = self.get_bounds()
        lb = np.concatenate((bounds.lb, np.array((0.001, -np.inf))))
        ub = (np.concatenate((bounds.ub, np.array((np.inf, np.inf)))),)
        keep_feasible = bounds.keep_feasible

        return SciOpt.Bounds(lb=lb, ub=ub, keep_feasible=keep_feasible)  # type: ignore

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
            "bounds": self.get_robust_bounds(),
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
        design = None
        solution = None

        default_kwargs = {
            "method": "SLSQP",
            "bounds": self.get_bounds(system=system),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

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


class QLSMSolver:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.cf = EndureCost(config)

    def kl_div_con(self, input):
        return np.exp(input) - 1

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
        h, T, Q, lamb, eta = x
        design = LSMDesign(h=h, T=T, Q=Q, policy=Policy.QFixed)
        query_cost = 0
        query_cost += z0 * self.kl_div_con((self.cf.Z0(design, system) - eta) / lamb)
        query_cost += z1 * self.kl_div_con((self.cf.Z1(design, system) - eta) / lamb)
        query_cost += q * self.kl_div_con((self.cf.Q(design, system) - eta) / lamb)
        query_cost += w * self.kl_div_con((self.cf.W(design, system) - eta) / lamb)
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
    ):
        h, T, Q = x
        design = LSMDesign(h=h, T=T, Q=Q, policy=Policy.QFixed)
        cost = self.cf.calc_cost(design, system, z0, z1, q, w)

        return cost

    def get_bounds(self, system: Optional[System] = None) -> SciOpt.Bounds:
        h_lb = self.config["lsm"]["bits_per_elem"]["min"]
        if system is None:
            h_ub = self.config["lsm"]["system"]["H"] - 0.1
        else:
            h_ub = system.H - 0.1
        t_ub = self.config["lsm"]["size_ratio"]["max"]
        t_lb = self.config["lsm"]["size_ratio"]["min"]
        q_ub = t_ub - 1
        q_lb = t_lb - 1

        lb = (h_lb, t_lb, q_lb)
        ub = (h_ub, t_ub, q_ub)
        keep_feasible = True

        # SciPy's typing with Bounds is not correct
        return SciOpt.Bounds(lb=lb, ub=ub, keep_feasible=keep_feasible)  # type: ignore

    def get_robust_bounds(self) -> SciOpt.Bounds:
        bounds = self.get_bounds()
        lb = np.concatenate((bounds.lb, np.array((0.001, -np.inf))))
        ub = (np.concatenate((bounds.ub, np.array((np.inf, np.inf)))),)
        keep_feasible = bounds.keep_feasible

        return SciOpt.Bounds(lb=lb, ub=ub, keep_feasible=keep_feasible)  # type: ignore

    def get_robust_design(
        self,
        system: System,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: np.ndarray = np.array(
            [H_DEFAULT, T_DEFAULT, Q_DEFAULT, LAMBDA_DEFAULT, ETA_DEFAULT]
        ),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        default_kwargs = {
            "method": "SLSQP",
            "bounds": self.get_robust_bounds(),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        solution = SciOpt.minimize(
            fun=lambda x: self.robust_objective(x, system, rho, z0, z1, q, w),
            x0=init_args,
            callback=callback_fn,
            **default_kwargs
        )
        design = LSMDesign(
            h=solution.x[0],
            T=solution.x[1],
            Q=solution.x[2],
            policy=Policy.QFixed
        )

        return design, solution

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
            "bounds": self.get_bounds(system=system),
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
            policy=Policy.QFixed,
        )

        return design, solution


