import os
import torch
from typing import Any

from torch.functional import Tensor
from torch.nn.parameter import Parameter

from endure.lcm.model.builder import LearnedCostModelBuilder


class LearnedCostModelLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any], model_path: str):
        super().__init__()
        # bpe_max = config["lsm"]["bits_per_elem"]["max"]
        # bpe_min = config["lsm"]["bits_per_elem"]["min"]
        # self._bpe_mean = Parameter(torch.Tensor([(bpe_max + bpe_min) / 2]))
        # self._bpe_std = Parameter(
        #     torch.sqrt(torch.Tensor([(bpe_max - bpe_min) ** 2 / 12]))
        # )
        self.normalize_bpe = config["ltune"]["data"]["normalize_inputs"]
        self.penalty_factor = config["ltune"]["penalty_factor"]
        self.mem_budget_idx = config["ltune"]["input_features"].index("H")

        self.lcm_builder = LearnedCostModelBuilder(config)
        self.model = self.lcm_builder.build_model()

        _, extension = os.path.splitext(model_path)
        is_checkpoint = extension == ".checkpoint"
        data = torch.load(os.path.join(config["io"]["data_dir"], model_path))
        if is_checkpoint:
            data = data["model_state_dict"]
        status = self.model.load_state_dict(data)
        assert len(status.missing_keys) == 0
        assert len(status.unexpected_keys) == 0

    def create_penalty_vector(self, bpe: Tensor, mem_budget: Tensor):
        penalty = torch.ones(bpe.size(dim=-1)).to(bpe.device)
        # for BPE guesses that exceed the maximum memory budget
        idx = bpe >= mem_budget
        penalty[idx] = self.penalty_factor * (bpe[idx] - mem_budget[idx])
        # for BPE guesses underneath 0
        idx = bpe < 0
        penalty[idx] = self.penalty_factor * bpe[idx] * -1

        return penalty

    def forward(self, pred, label):
        # For learned cost model loss, pred is the DB configuration, label is
        # the workload

        bpe = pred[:, 0]
        mem_budget = label[:, self.mem_budget_idx]
        penalty = self.create_penalty_vector(bpe, mem_budget)

        bpe = bpe.view(-1, 1)
        size_ratio = torch.argmax(pred[:, 1:], dim=-1).view(-1, 1)
        # if self.normalize_bpe:
        #     bpe = ((bpe - self._bpe_mean) / self._bpe_std)

        inputs = torch.concat([label, bpe, size_ratio], dim=-1)
        out = self.model(inputs)
        out = out.sum(dim=-1)
        out = out * penalty.view(-1, 1)

        return out.mean()
