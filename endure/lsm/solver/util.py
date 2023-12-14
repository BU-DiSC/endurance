from typing import Any, Optional, Callable, Tuple

import numpy as np
import scipy.optimize as SciOpt

from endure.lsm.types import Policy, System

H_DEFAULT = 3
T_DEFAULT = 3
Z_DEFAULT = 1
Y_DEFAULT = 1
Q_DEFAULT = 1
K_DEFAULT = 1
LAMBDA_DEFAULT = 1
ETA_DEFAULT = 1

def kl_div_con(input: float):
    return np.exp(input) - 1

def get_t_bounds(config: dict) -> Tuple:
    t_ub = config["lsm"]["size_ratio"]["max"]
    t_lb = config["lsm"]["size_ratio"]["min"]

    return (t_lb, t_ub)

def get_h_bounds(config: dict, system: Optional[System] = None) -> Tuple:
    h_lb = config["lsm"]["bits_per_elem"]["min"]
    if system is None:
        h_ub = config["lsm"]["system"]["H"] - 0.1
    else:
        h_ub = system.H - 0.1

    return (h_lb, h_ub)

def get_lambda_bounds() -> Tuple:
    return (0.001, np.inf)

def get_eta_bounds() -> Tuple:
    return (-np.inf, np.inf)

def get_bounds(
    config: dict,
    policy: Policy = Policy.Leveling,
    system: Optional[System] = None,
    robust: bool = False
) -> SciOpt.Bounds:
    t_bounds = get_t_bounds(config)
    h_bounds = get_h_bounds(config, system)

    lb = (h_bounds[0], t_bounds[0])
    ub = (h_bounds[1], t_bounds[1])
    if policy == Policy.QFixed:
        lb += (t_bounds[0] - 1,)
        ub += (t_bounds[1] - 1,)
    elif policy == Policy.YZHybrid:
        lb += (t_bounds[0] - 1, t_bounds[0] - 1)
        ub += (t_bounds[1] - 1, t_bounds[1] - 1)
    elif policy == Policy.KHybrid:
        max_levels: int = config["lsm"]["max_levels"]
        lb += tuple(t_bounds[0] - 1 for _ in range(max_levels))
        ub += tuple(t_bounds[1] - 1 for _ in range(max_levels))
    elif policy in (Policy.Tiering, Policy.Leveling):
        pass # No need to add more items for classic policy

    if robust:
        lambda_bounds = get_lambda_bounds()
        eta_bounds = get_eta_bounds()
        lb += (lambda_bounds[0], eta_bounds[0])
        ub += (lambda_bounds[1], eta_bounds[1])

    return SciOpt.Bounds(lb=lb, ub=ub, keep_feasible=True)  # type: ignore
