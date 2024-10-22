import numpy as np
import axe.lsm.lsm_cost as Cost
from axe.lsm.types import Policy, System, LSMDesign


class EndureCost:
    def __init__(self, max_levels: int) -> None:
        super().__init__()
        self.max_levels = max_levels

    def L(self, design: LSMDesign, system: System, ceil=False):
        level = Cost.calc_level(design.h, design.T, system.E, system.H, system.N, ceil)

        return level

    def mbuff(self, design: LSMDesign, system: System):
        return Cost.calc_mbuff(design.h, system.H, system.N)

    def create_k_list(self, design: LSMDesign, system: System) -> np.ndarray:
        if design.policy is Policy.KHybrid:
            assert design.K is not None
            k = np.array(design.K)
        elif design.policy is Policy.Tiering:
            k = np.full(self.max_levels, design.T - 1)
        elif design.policy is Policy.Leveling:
            k = np.ones(self.max_levels)
        elif design.policy is Policy.YZHybrid:
            levels = Cost.calc_level(
                design.h, design.T, system.E, system.H, system.N, True
            )
            levels = int(levels)
            k = np.full(levels - 1, design.Y)
            k = np.concatenate((k, [design.Z]))
            k = np.pad(
                k,
                (0, self.max_levels - len(k)),
                "constant",
                constant_values=(1.0, 1.0),
            )
        elif design.policy is Policy.QFixed:
            k = np.full(self.max_levels, design.Q)
        else:
            k = np.ones(self.max_levels)

        return k

    def Z0(self, design: LSMDesign, system: System) -> float:
        k = self.create_k_list(design, system)
        cost = Cost.empty_op(design.h, design.T, k, system.N, system.E, system.H)

        return cost

    def Z1(self, design: LSMDesign, system: System) -> float:
        k = self.create_k_list(design, system)
        cost = Cost.non_empty_op(design.h, design.T, k, system.E, system.H, system.N)

        return cost

    def Q(self, design: LSMDesign, system: System) -> float:
        k = self.create_k_list(design, system)
        cost = Cost.range_op(
            design.h, design.T, k, system.B, system.s, system.E, system.H, system.N
        )

        return cost

    def W(self, design: LSMDesign, system: System) -> float:
        k = self.create_k_list(design, system)
        cost = Cost.write_op(
            design.h, design.T, k, system.B, system.E, system.H, system.N, system.phi
        )

        return cost

    def calc_cost(
        self,
        design: LSMDesign,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ):
        k = self.create_k_list(design, system)
        cost = Cost.calc_cost(
            design.h,
            design.T,
            k,
            z0,
            z1,
            q,
            w,
            system.B,
            system.s,
            system.E,
            system.H,
            system.N,
            system.phi,
        )

        return cost
