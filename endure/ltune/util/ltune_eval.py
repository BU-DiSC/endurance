from typing import Any

from torch import Tensor
import torch

from endure.lcm.util import eval_lcm_impl
from endure.lsm.types import LSMDesign, System, Policy
from endure.ltune.data.generator import LTuneGenerator
from endure.ltune.loss import LearnedCostModelLoss


class LTuneEvalUtil:
    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module,
    ) -> None:
        self.gen = LTuneGenerator(config)
        self.loss = LearnedCostModelLoss(
            config,
            config["job"]["LTuneTrain"]["loss_fn_path"]
        )
        self.max_t = config["lsm"]["size_ratio"]["max"]
        self.min_t = config["lsm"]["size_ratio"]["min"]
        self.model = model
        self.config = config

    def calc_size_ratio_range(self) -> int:
        return self.max_t - self.min_t + 1

    def eval_lcm(
        self,
        design: LSMDesign,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        return eval_lcm_impl(design, system, z0, z1, q, w,
                             self.model, self.min_t, self.max_t)

    def eval_ltune(
        self,
        design: LSMDesign,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:

        return 0

    def convert_ltune_output(self, output: Tensor, design_choice: str):
        if design_choice == "QLSM":
            design = self._qlsm_convert(output)
        else:
            design = self._classic_convert(output)

        return design

    def _qlsm_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.calc_size_ratio_range()
        h = out[0].item()
        t = torch.argmax(out[1:cap_range]).item() + 2
        q = torch.argmax(out[cap_range:]).item() + 1

        return LSMDesign(h=h, T=t, Q=q, policy=Policy.QFixed)

    def _classic_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.calc_size_ratio_range()
        h = out[0].item()
        t = torch.argmax(out[1:cap_range]).item() + 2
        policy_val = torch.argmax(out[cap_range:]).item()
        if policy_val:
            policy = Policy.Leveling
        else:
            policy = Policy.Tiering

        return LSMDesign(h=h, T=t, policy=policy)

