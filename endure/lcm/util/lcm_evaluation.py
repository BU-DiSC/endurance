from typing import Any, Self

from torch import Tensor
import torch

from .util import one_hot_lcm, one_hot_lcm_classic
from endure.lcm.data.generator import LCMDataGenerator
from endure.lsm.cost import EndureCost
from endure.lsm.types import LSMDesign, Policy, System

class LCMEvalUtil:
    def __init__(
        self: Self,
        config: dict[str, Any],
        model: torch.nn.Module,
        generator: LCMDataGenerator,
    ) -> None:
        self.config = config
        self.model = model
        self.gen = generator
        self.cf = EndureCost(config)

    def eval_lcm(
        self: Self,
        design: LSMDesign,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        x = self.create_input_from_types(design, system, z0, z1, q, w)
        x = x.to(torch.float).view(1, -1)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
            pred = pred.sum().item()

        return pred

    def create_input_from_types(
        self: Self,
        design: LSMDesign,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> Tensor:
        categories = (self.config['lsm']['size_ratio']['max'] -
                      self.config['lsm']['size_ratio']['min'] + 1)
        wl = [z0, z1, q, w]
        sys = [system.B, system.s, system.E, system.H, system.N]
        if design.policy in (Policy.Tiering, Policy.Leveling):
            inputs = wl + sys + [design.h, design.T, design.policy.value]
            data = torch.Tensor(inputs)
            out = one_hot_lcm_classic(torch.Tensor(inputs), categories)
        else: # design.policy == Policy.QFixed
            inputs = wl + sys + [design.h, design.T, design.Q]
            data = torch.Tensor(inputs)
            out = one_hot_lcm(data, len(inputs), 2, categories)

        return out

    def gen_random_sample(self):
        row = {}
        z0, z1, q, w = self.gen._sample_workload(4)
        system = self.gen._sample_system()
        design = self.gen._sample_design(system)
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
        row['cost_lcm'] = cost_lcm
        row['cost_acm'] = cost_acm

        return row


