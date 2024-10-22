import torch

from .util import eval_lcm_impl
from axe.lcm.data.generator import LCMDataGenerator
from axe.lsm.cost import EndureCost
from axe.lsm.types import LSMDesign, Policy, System

class LCMEvalUtil:
    def __init__(
        self,
        min_size_ratio: int,
        max_size_ratio: int,
        max_levels: int,
        model: torch.nn.Module,
        generator: LCMDataGenerator,
    ) -> None:
        self.cf = EndureCost(max_levels=max_levels)

        self.min_t = min_size_ratio
        self.max_t = max_size_ratio
        self.model = model
        self.gen = generator

    def eval_lcm(
        self,
        design: LSMDesign,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        return eval_lcm_impl(design, system, z0, z1, q, w,
                             self.model, self.min_t, self.max_t)

    def gen_random_sample(self):
        row = {}
        z0, z1, q, w = self.gen._sample_workload(4)
        system: System = self.gen._sample_system()
        design: LSMDesign = self.gen._sample_design(system)
        cost_lcm = self.eval_lcm(design, system, z0, z1, q, w)
        cost_acm = self.cf.calc_cost(design, system, z0, z1, q, w)
        row = {
            "z0": z0,
            "z1": z1,
            "q": q,
            "w": w,
            "B": system.B,
            "s": system.s,
            "E": system.E,
            "H": system.H,
            "N": system.N,
            'h': design.h,
            'T': design.T,
        }
        if design.policy in (Policy.Tiering, Policy.Leveling):
            row["policy"] = design.policy.value
        elif design.policy == Policy.KHybrid:
            for idx, k in enumerate(design.K):
                row[f"K_{idx}"] = k
        elif design.policy == Policy.QFixed:
            row['Q'] = design.Q
        elif design.policy == Policy.YZHybrid:
            row['Y'] = design.Y
            row['Z'] = design.Z
        row['cost_lcm'] = cost_lcm
        row['cost_acm'] = cost_acm

        return row, design, system
        
