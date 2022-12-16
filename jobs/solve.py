from typing import Optional, Callable
import lsm.cost as CostFunc
import numpy as np
import scipy.optimize as SciOpt
from lsm.lsmtype import Policy


class EndureSolver:
    def __init__(self, config: dict):
        self._config = config
        self._cf = None

    def kl_div_con(self, input):
        return np.exp(input) - 1

    def z0_conjugate(self, x):
        pass

    def z1_conjugate(self, x):
        pass

    def q_conjugate(self, x):
        pass

    def w_conjugate(self, x):
        pass

    def robust_objective(
        self,
        x: list,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        eta = x[-1]
        lamb = x[-2]
        query_cost = ((z0 * self.z0_conjugate(x))
                      + (z1 * self.z1_conjugate(x))
                      + (q * self.q_conjugate(x))
                      + (w * self.w_conjugate(x)))
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
        pass

    def solve_nominal(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: list,
        bounds: SciOpt.Bounds,
        callback_fun: Optional[Callable[..., float]] = None
    ):
        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'options': {'ftol': 1e-6, 'disp': False}}

        sol = SciOpt.minimize(
            fun=lambda x: self.nominal_objective(x, z0, z1, q, w),
            x0=init_args,
            callback=callback_fun,
            **minimizer_kwargs)
        return sol

    def solve_robust(
        self,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: list,
        bounds: SciOpt.Bounds,
        callback_fun: Optional[Callable[..., float]] = None
    ):
        init_args = init_args + [1., 1.]  # manually add lambda and eta
        bounds = SciOpt.Bounds(
            np.concatenate([bounds.lb, np.array([0.01, -np.inf])]),
            np.concatenate([bounds.ub, np.array([np.inf, np.inf])]),
            keep_feasible=bounds.keep_feasible)

        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'options': {'ftol': 1e-12, 'disp': False}}

        sol = SciOpt.minimize(
            fun=lambda x: self.robust_objective(x, rho, z0, z1, q, w),
            x0=init_args,
            callback=callback_fun,
            **minimizer_kwargs)
        return sol


class EndureTierLevelSolver(EndureSolver):
    def __init__(self, config: dict, policy: Policy):
        super().__init__(config)
        self._cf = CostFunc.EndureTierLevelCost(**config['system'])
        self.policy = policy

    def z0_conjugate(self, x: list):
        h, T, lamb, eta = x
        kl_conjugate_input = (self._cf.Z0(h, T, self.policy) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def z1_conjugate(self, x: list):
        h, T, lamb, eta = x
        kl_conjugate_input = (self._cf.Z1(h, T, self.policy) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def q_conjugate(self, x: list):
        h, T, lamb, eta = x
        kl_conjugate_input = (self._cf.Q(h, T, self.policy) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def w_conjugate(self, x: list):
        h, T, lamb, eta = x
        kl_conjugate_input = (self._cf.W(h, T, self.policy) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

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

    def get_bounds(self):
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
        h_init: float = 1.0,
        t_init: float = 2.0,
    ):
        init_args = [h_init, t_init]
        bounds = self.get_bounds()
        sol = self.solve_robust(rho, z0, z1, q, w, init_args, bounds)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        h_init: float = 1.0,
        t_init: float = 2.0,
    ):
        init_args = [h_init, t_init]
        bounds = self.get_bounds()
        sol = self.solve_nominal(z0, z1, q, w, init_args, bounds)
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

    def z0_conjugate(self, x: list):
        h, T, Q, lamb, eta = x
        kl_conjugate_input = (self._cf.Z0(h, T, Q) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def z1_conjugate(self, x: list):
        h, T, Q, lamb, eta = x
        kl_conjugate_input = (self._cf.Z1(h, T, Q) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def q_conjugate(self, x: list):
        h, T, Q, lamb, eta = x
        kl_conjugate_input = (self._cf.Q(h, T, Q) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def w_conjugate(self, x: list):
        h, T, Q, lamb, eta = x
        kl_conjugate_input = (self._cf.W(h, T, Q) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

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
        h_init: float = 1.0,
        t_init: float = 2.0,
        q_init: float = 1.0,
    ):
        init_args = [h_init, t_init, q_init]
        bounds = self.get_bounds()
        sol = self.solve_robust(rho, z0, z1, q, w, init_args, bounds)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        h_init: float = 1.0,
        t_init: float = 2.0,
        q_init: float = 1.0,
    ):
        init_args = [h_init, t_init, q_init]
        bounds = self.get_bounds()
        sol = self.solve_nominal(z0, z1, q, w, init_args, bounds)
        return sol


class RobustKSolver(EndureSolver):
    def __init__(self, config: dict):
        super().__init__(config)
        self._cf = CostFunc.EndureKHybridCost(**config['system'])

    def z0_conjugate(self, x: list):
        lamb, eta = x[-2:]
        h, T = x[0:2]
        K = x[2:-2]
        kl_conjugate_input = (self._cf.Z0(h, T, K) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def z1_conjugate(self, x: list):
        lamb, eta = x[-2:]
        h, T = x[0:2]
        K = x[2:-2]
        kl_conjugate_input = (self._cf.Z1(h, T, K) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def q_conjugate(self, x: list):
        lamb, eta = x[-2:]
        h, T = x[0:2]
        K = x[2:-2]
        kl_conjugate_input = (self._cf.Q(h, T, K) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

    def w_conjugate(self, x: list):
        lamb, eta = x[-2:]
        h, T = x[0:2]
        K = x[2:-2]
        kl_conjugate_input = (self._cf.W(h, T, K) - eta) / lamb
        return self.kl_div_con(kl_conjugate_input)

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
        h_init: float = 5.0,
        t_init: float = 10.0,
        k_inits: float = 1.0,
    ):
        MAX_LEVELS = self._config['lsm']['max_levels']
        init_args = [h_init, t_init] + [k_inits] * MAX_LEVELS
        bounds = self.get_bounds()
        sol = self.solve_robust(rho, z0, z1, q, w, init_args, bounds)
        return sol

    def find_nominal_design(
        self,
        z0: float,
        z1: float,
        q: float,
        w: float,
        h_init: float = 5.0,
        t_init: float = 10.0,
        k_inits: float = 1.0,
    ):
        MAX_LEVELS = self._config['lsm']['max_levels']
        init_args = [h_init, t_init] + [k_inits] * MAX_LEVELS
        bounds = self.get_bounds()
        sol = self.solve_nominal(z0, z1, q, w, init_args, bounds)
        return sol
