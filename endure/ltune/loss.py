from typing import Any
import os
import torch

from torch.functional import Tensor
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

    def create_penalty_vector(
        self, bpe: Tensor, mem_budget: Tensor, size_ratio: Tensor
    ):
        penalty = torch.ones(bpe.size()).to(bpe.device)
        # for BPE guesses that exceed the maximum memory budget
        idx = bpe >= mem_budget
        penalty[idx] = self.penalty_factor * (bpe[idx] - mem_budget[idx])
        # for BPE guesses underneath 0
        idx = bpe < 0
        penalty[idx] = self.penalty_factor * (0 - bpe[idx])

        penalty[size_ratio > 48] = self.penalty_factor * (
            size_ratio[size_ratio > 48] - 48
        )
        penalty[size_ratio < 0] = self.penalty_factor * (0 - size_ratio[size_ratio < 0])

        return penalty

    def convert_tuner_output(self, tuner_out):
        bpe = tuner_out[:, 0]
        bpe = bpe.view(-1, 1)
        # size_ratio = torch.argmax(tuner_out[:, 1:], dim=-1).view(-1, 1)
        size_ratio = torch.ceil(tuner_out[:, 1:] - 2).view(-1, 1)

        return bpe, size_ratio

    def forward(self, pred, label):
        # For learned cost model loss, pred is the DB configuration, label is
        # the workload
        bpe, size_ratio = self.convert_tuner_output(pred)
        penalty = self.create_penalty_vector(
            bpe, label[:, self.mem_budget_idx].view(-1, 1), size_ratio
        )
        size_ratio[size_ratio > 48] = 48
        size_ratio[size_ratio < 0] = 0
        # if self.normalize_bpe:
        #     bpe = ((bpe - self._bpe_mean) / self._bpe_std)

        inputs = torch.concat([label, bpe, size_ratio], dim=-1)
        out = self.model(inputs)
        out = out.sum(dim=-1)
        out = out * penalty

        return out.mean()
