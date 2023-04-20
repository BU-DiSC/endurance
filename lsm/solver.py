from typing import Optional, Callable
import lsm.cost as CostFunc
import numpy as np
import scipy.optimize as SciOpt
from lsm.lsmtype import Policy

H_DEFAULT = 4
T_DEFAULT = 10
Z_DEFAULT = 8
Y_DEFAULT = 8
Q_DEFAULT = 8
K_DEFAULT = 8
LAMBDA_DEFAULT = 1
ETA_DEFAULT = 1
SOLVER_FTOL = 1e-12


class EndureSolver:
    def __init__(self, config: dict):
        self._config = config
        self._cf = None

    def kl_div_con(self, input):
        return np.exp(input) - 1

    def robust_objective(
        self,
        x: list,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        raise NotImplementedError

    def nominal_objective(
        self,
        x: list,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        raise NotImplementedError

    def _solve_nominal(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: list,
        bounds: SciOpt.Bounds,
        callback_fun: Optional[Callable[..., float]] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        min_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'options': {'ftol': SOLVER_FTOL, 'disp': False}}
        if minimizer_kwargs is not None:
            min_kwargs.update(minimizer_kwargs)

        sol = SciOpt.minimize(
            fun=lambda x: self.nominal_objective(x, z0, z1, q, w),
            x0=init_args,
            callback=callback_fun,
            **min_kwargs)

        return sol

    def _solve_robust(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: list,
        bounds: SciOpt.Bounds,
        callback_fun: Optional[Callable[..., float]] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        init_args = init_args + [LAMBDA_DEFAULT, ETA_DEFAULT]
        bounds = SciOpt.Bounds(
            np.concatenate([bounds.lb, np.array([0.01, -np.inf])]),
            np.concatenate([bounds.ub, np.array([np.inf, np.inf])]),
            keep_feasible=bounds.keep_feasible)

        min_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'options': {'ftol': SOLVER_FTOL, 'disp': False}}
        if minimizer_kwargs is not None:
            min_kwargs.update(minimizer_kwargs)

        sol = SciOpt.minimize(
            fun=lambda x: self.robust_objective(x, rho, z0, z1, q, w),
            x0=init_args,
            callback=callback_fun,
            **min_kwargs)

        return sol

    def find_robust_design(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        raise NotImplementedError

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        raise NotImplementedError


class EndureTierLevelSolver(EndureSolver):
    def __init__(self, config: dict, policy: Policy):
        super().__init__(config)
        self._cf = CostFunc.EndureTierLevelCost(**config['system'])
        self.policy = policy

    def robust_objective(
        self,
        x: list,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, lamb, eta = x
        query_cost = z0 * \
            self.kl_div_con((self._cf.Z0(h, T, self.policy) - eta) / lamb)
        query_cost += z1 * \
            self.kl_div_con((self._cf.Z1(h, T, self.policy) - eta) / lamb)
        query_cost += q * \
            self.kl_div_con((self._cf.Q(h, T, self.policy) - eta) / lamb)
        query_cost += w * \
            self.kl_div_con((self._cf.W(h, T, self.policy) - eta) / lamb)
        cost = eta + (rho * lamb) + (lamb * query_cost)
        return cost

    def nominal_objective(
        self,
        x: list,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T = x
        return self._cf.calc_cost(h, T, self.policy, z0, z1, q, w)

    def get_bounds(self) -> SciOpt.Bounds:
        T_UPPER_LIM = self._config['lsm']['size_ratio']['max']
        T_LOWER_LIM = self._config['lsm']['size_ratio']['min']
        H_LOWER_LIM = self._config['lsm']['bits_per_elem']['min']
        H_UPPER_LIM = self._config['lsm']['bits_per_elem']['max']

        return SciOpt.Bounds(
            [H_LOWER_LIM, T_LOWER_LIM],
            [H_UPPER_LIM, T_UPPER_LIM],
            keep_feasible=True)

    def find_robust_design(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_robust(
            rho, z0, z1, q, w, init_args, bounds, minimizer_kwargs)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_nominal(
            z0, z1, q, w, init_args, bounds, minimizer_kwargs)
        return sol


class EndureLevelSolver(EndureTierLevelSolver):
    def __init__(self, config: dict):
        super().__init__(config, Policy.Leveling)


class EndureTierSolver(EndureTierLevelSolver):
    def __init__(self, config: dict):
        super().__init__(config, Policy.Tiering)


class EndureQSolver(EndureSolver):
    def __init__(self, config: dict):
        super().__init__(config)
        self._cf = CostFunc.EndureQFixedCost(**config['system'])

    def robust_objective(
        self,
        x: list,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, Q, lamb, eta = x
        query_cost = z0 * self.kl_div_con((self._cf.Z0(h, T, Q) - eta) / lamb)
        query_cost += z1 * self.kl_div_con((self._cf.Z1(h, T, Q) - eta) / lamb)
        query_cost += q * self.kl_div_con((self._cf.Q(h, T, Q) - eta) / lamb)
        query_cost += w * self.kl_div_con((self._cf.W(h, T, Q) - eta) / lamb)
        cost = eta + (rho * lamb) + (lamb * query_cost)
        return cost

    def nominal_objective(
        self,
        x: list,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, Q = x
        return self._cf.calc_cost(h, T, Q, z0, z1, q, w)

    def get_bounds(self):
        T_UPPER_LIM = self._config['lsm']['size_ratio']['max']
        T_LOWER_LIM = self._config['lsm']['size_ratio']['min']
        H_LOWER_LIM = self._config['lsm']['bits_per_elem']['min']
        H_UPPER_LIM = self._config['lsm']['bits_per_elem']['max']

        return SciOpt.Bounds(
            [H_LOWER_LIM, T_LOWER_LIM, T_LOWER_LIM - 1],
            [H_UPPER_LIM, T_UPPER_LIM, T_UPPER_LIM - 1],
            keep_feasible=True)

    def find_robust_design(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT, Q_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_robust(rho, z0, z1, q, w, init_args, bounds,
                                 minimizer_kwargs=minimizer_kwargs)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT, Q_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_nominal(z0, z1, q, w, init_args, bounds,
                                  minimizer_kwargs=minimizer_kwargs)
        return sol


class EndureKSolver(EndureSolver):
    def __init__(self, config: dict):
        super().__init__(config)
        self._cf = CostFunc.EndureKHybridCost(**config['system'])

    def robust_objective(
        self,
        x: list,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        lamb, eta = x[-2:]
        h, T = x[0:2]
        K = x[2:-2]
        query_cost = z0 * self.kl_div_con((self._cf.Z0(h, T, K) - eta) / lamb)
        query_cost += z1 * self.kl_div_con((self._cf.Z1(h, T, K) - eta) / lamb)
        query_cost += q * self.kl_div_con((self._cf.Q(h, T, K) - eta) / lamb)
        query_cost += w * self.kl_div_con((self._cf.W(h, T, K) - eta) / lamb)
        cost = eta + (rho * lamb) + (lamb * query_cost)
        return cost

    def nominal_objective(
        self,
        x: list,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T = x[0:2]
        K = x[2:]
        return self._cf.calc_cost(h, T, K, z0, z1, q, w)

    def get_bounds(self):
        T_UPPER_LIM = self._config['lsm']['size_ratio']['max']
        T_LOWER_LIM = self._config['lsm']['size_ratio']['min']
        H_LOWER_LIM = self._config['lsm']['bits_per_elem']['min']
        H_UPPER_LIM = self._config['lsm']['bits_per_elem']['max']
        MAX_LEVELS = self._config['lsm']['max_levels']

        k_lower = [T_LOWER_LIM - 1] * MAX_LEVELS
        k_upper = [T_UPPER_LIM - 1] * MAX_LEVELS

        return SciOpt.Bounds(
            [H_LOWER_LIM, T_LOWER_LIM] + k_lower,
            [H_UPPER_LIM, T_UPPER_LIM] + k_upper,
            keep_feasible=True)

    def find_robust_design(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT]
            init_args += [K_DEFAULT] * self._config['lsm']['max_levels']
        bounds = self.get_bounds()
        sol = self._solve_robust(rho, z0, z1, q, w, init_args, bounds,
                                 minimizer_kwargs=minimizer_kwargs)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT]
            init_args += [K_DEFAULT] * self._config['lsm']['max_levels']
        bounds = self.get_bounds()
        sol = self._solve_nominal(z0, z1, q, w, init_args, bounds,
                                  minimizer_kwargs=minimizer_kwargs)
        return sol


class EndureYZSolver(EndureSolver):
    def __init__(self, config: dict):
        super().__init__(config)
        self._cf = CostFunc.EndureYZHybridCost(**config['system'])

    def robust_objective(
        self,
        x: list,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, Y, Z, lamb, eta = x
        query_cost = z0 * \
            self.kl_div_con((self._cf.Z0(h, T, Y, Z) - eta) / lamb)
        query_cost += z1 * \
            self.kl_div_con((self._cf.Z1(h, T, Y, Z) - eta) / lamb)
        query_cost += q * self.kl_div_con((self._cf.Q(h, T, Y, Z) - eta) / lamb)
        query_cost += w * self.kl_div_con((self._cf.W(h, T, Y, Z) - eta) / lamb)
        cost = eta + (rho * lamb) + (lamb * query_cost)
        return cost

    def nominal_objective(
        self,
        x: list,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, Y, Z = x
        return self._cf.calc_cost(h, T, Y, Z, z0, z1, q, w)

    def get_bounds(self):
        T_UPPER_LIM = self._config['lsm']['size_ratio']['max']
        T_LOWER_LIM = self._config['lsm']['size_ratio']['min']
        H_LOWER_LIM = self._config['lsm']['bits_per_elem']['min']
        H_UPPER_LIM = self._config['lsm']['bits_per_elem']['max']

        return SciOpt.Bounds(
            [H_LOWER_LIM, T_LOWER_LIM, T_LOWER_LIM - 1, T_LOWER_LIM - 1],
            [H_UPPER_LIM, T_UPPER_LIM, T_UPPER_LIM - 1, T_UPPER_LIM - 1],
            keep_feasible=True)

    def find_robust_design(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT, Y_DEFAULT, Z_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_robust(rho, z0, z1, q, w, init_args, bounds,
                                 minimizer_kwargs=minimizer_kwargs)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT, Y_DEFAULT, Z_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_nominal(z0, z1, q, w, init_args, bounds,
                                  minimizer_kwargs=minimizer_kwargs)
        return sol


class EndureYSolver(EndureSolver):
    def __init__(self, config: dict, last_level: float):
        super().__init__(config)
        self._cf = CostFunc.EndureYZHybridCost(**config['system'])
        self.Z = last_level

    def robust_objective(
        self,
        x: list,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, Y, lamb, eta = x
        query_cost = z0 * \
            self.kl_div_con((self._cf.Z0(h, T, Y, self.Z) - eta) / lamb)
        query_cost += z1 * \
            self.kl_div_con((self._cf.Z1(h, T, Y, self.Z) - eta) / lamb)
        query_cost += q * \
            self.kl_div_con((self._cf.Q(h, T, Y, self.Z) - eta) / lamb)
        query_cost += w * \
            self.kl_div_con((self._cf.W(h, T, Y, self.Z) - eta) / lamb)
        cost = eta + (rho * lamb) + (lamb * query_cost)
        return cost

    def nominal_objective(
        self,
        x: list,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        h, T, Y = x
        return self._cf.calc_cost(h, T, Y, self.Z, z0, z1, q, w)

    def get_bounds(self):
        T_UPPER_LIM = self._config['lsm']['size_ratio']['max']
        T_LOWER_LIM = self._config['lsm']['size_ratio']['min']
        H_LOWER_LIM = self._config['lsm']['bits_per_elem']['min']
        H_UPPER_LIM = self._config['lsm']['bits_per_elem']['max']

        return SciOpt.Bounds(
            [H_LOWER_LIM, T_LOWER_LIM, T_LOWER_LIM - 1],
            [H_UPPER_LIM, T_UPPER_LIM, T_UPPER_LIM - 1],
            keep_feasible=True)

    def find_robust_design(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT, Y_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_robust(rho, z0, z1, q, w, init_args, bounds,
                                 minimizer_kwargs=minimizer_kwargs)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: Optional[list] = None,
        minimizer_kwargs: Optional[dict] = None,
    ) -> SciOpt.OptimizeResult:
        if init_args is None:
            init_args = [H_DEFAULT, T_DEFAULT, Y_DEFAULT]
        bounds = self.get_bounds()
        sol = self._solve_nominal(z0, z1, q, w, init_args, bounds,
                                  minimizer_kwargs=minimizer_kwargs)
        return sol