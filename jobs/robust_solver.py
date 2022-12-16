import lsm.cost as CostFunc
import numpy as np
import scipy.optimize as SciOpt


class CreateQRobustDesign:
    def __init__(self, config: dict):
        self._config = config
        self._cf = CostFunc.EndureQFixedCost(**config['system'])
        self.rho = 0.

    def kl_div_con(self, s):
        return np.exp(s) - 1

    def calculate_objective(
        self,
        args: list,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        h, T, Q, lamb, eta = args

        total_cost = 0
        total_cost += z0 * self.kl_div_con((self._cf.Z0(h, T, Q) - eta) / lamb)
        total_cost += z1 * self.kl_div_con((self._cf.Z1(h, T, Q) - eta) / lamb)
        total_cost += q * self.kl_div_con((self._cf.Q(h, T, Q) - eta) / lamb)
        total_cost += w * self.kl_div_con((self._cf.W(h, T, Q) - eta) / lamb)
        cost = eta + (self.rho * lamb) + (lamb * total_cost)
        return cost

    def cf_callback(self, x):
        h, T, Q, eta, lamb = x
        z0 = self.kl_div_con((self._cf.Z0(h, T, Q) - eta) / lamb)
        z1 = self.kl_div_con((self._cf.Z1(h, T, Q) - eta) / lamb)
        q = self.kl_div_con((self._cf.Q(h, T, Q) - eta) / lamb)
        w = self.kl_div_con((self._cf.W(h, T, Q) - eta) / lamb)

        print(f'{eta:.2f}'
              f'\t {lamb:.2f}'
              f'\t {z0:.2f}'
              f'\t {z1:.2f}'
              f'\t {q:.2f}'
              f'\t {w:.2f}'
              f'\t {self.calculate_objective(x):.6f}'
              f'\t {h:.6f}'
              f'\t {T:.6f}')

    def get_robust_leveling_design(
        self,
        rho: float,
        z0: float = 0.,
        z1: float = 0.,
        w: float = 0.,
        q: float = 0.,
        nominal_design: dict = None
    ):
        self.rho = rho
        T_UPPER_LIM = self._config['lsm']['size_ratio']['max']
        T_LOWER_LIM = self._config['lsm']['size_ratio']['min']
        H_LOWER_LIM = self._config['lsm']['bits_per_elem']['min']
        H_UPPER_LIM = self._config['lsm']['bits_per_elem']['max']
        h_init = 1.
        t_init = 5.
        q_init = 5.

        # design = {}
        # Check leveling cost
        bounds = SciOpt.Bounds(
            [H_LOWER_LIM, T_LOWER_LIM, T_LOWER_LIM - 1, 0.1, -np.inf],
            [H_UPPER_LIM, T_UPPER_LIM, T_UPPER_LIM - 1, np.inf, np.inf],
            keep_feasible=True)

        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'options': {'ftol': 1e-12, 'disp': False}}

        self._cf.is_leveling_policy = True
        sol = SciOpt.minimize(fun=self.calculate_objective,
                              x0=np.array([h_init, t_init, q_init, 1., 1.]),
                              #    callback = self.cf_callback,
                              **minimizer_kwargs)
        # cost = self._cf.calculate_cost(sol.x[0], sol.x[1])
        # design['exit_mode'] = sol.status
        # design['T'] = sol.x[1]
        # design['M_filt'] = sol.x[0] * self._cf.N
        # design['M_buff'] = self._cf.M - design['M_filt']
        # design['is_leveling_policy'] = True
        # design['lambda'] = sol.x[2]
        # design['eta'] = sol.x[3]
        # design['cost'] = cost
        # design['obj'] = sol.fun
        return sol
