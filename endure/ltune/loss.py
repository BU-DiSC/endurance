from typing import Any
import os

import torch
import toml

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
        self.categorical = config["ltune"]["model"]["categorical"]

        self._lcm_config = toml.load(
            os.path.join(config["io"]["data_dir"], model_path, "endure.toml")
        )
        self.lcm_builder = LearnedCostModelBuilder(self._lcm_config)
        self.model = self.lcm_builder.build_model()

        data = torch.load(
            os.path.join(config["io"]["data_dir"], model_path, "best.model")
        )
        status = self.model.load_state_dict(data)
        assert len(status.missing_keys) == 0
        assert len(status.unexpected_keys) == 0
        self.model.eval()

    def create_penalty_vector(self, bpe: torch.Tensor, mem_budget: torch.Tensor):
        penalty = torch.zeros(bpe.size()).to(bpe.device)
        # for BPE guesses that exceed the maximum memory budget
        idx = bpe >= mem_budget
        penalty[idx] = self.penalty_factor * (bpe[idx] - mem_budget[idx])
        # for BPE guesses underneath 0
        idx = bpe < 0
        penalty[idx] = self.penalty_factor * (0 - bpe[idx])

        return penalty

    def split_tuner_out(self, tuner_out):
        bpe = tuner_out[:, 0]
        bpe = bpe.view(-1, 1)
        size_ratio = tuner_out[:, 1:]

        return bpe, size_ratio

    def forward(self, pred, label):
        assert self.model.training is False
        # For learned cost model loss, pred is the DB configuration, label is
        # the workload
        bpe, size_ratio = self.split_tuner_out(pred)
        mem_budget = label[:, self.mem_budget_idx].view(-1, 1)
        penalty = self.create_penalty_vector(bpe, mem_budget)
        bpe[bpe > mem_budget] = mem_budget[bpe > mem_budget]
        bpe[bpe < 0] = 0

        inputs = torch.concat([label, bpe, size_ratio], dim=-1)
        out = self.model(inputs)
        out = out.sum(dim=-1)
        out = out + penalty
        out = out.mean()

        return out
