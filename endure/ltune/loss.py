from typing import Any
import os

import torch
import toml

from endure.lcm.model import LearnedCostModelBuilder


class LearnedCostModelLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any], model_path: str):
        super().__init__()
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
        self.model.eval()

    def calc_mem_penalty(self, label, bpe):
        mem_budget = label[:, self.mem_budget_idx].view(-1, 1)
        penalty = torch.zeros(bpe.size()).to(bpe.device)
        idx = bpe >= mem_budget
        penalty[idx] = self.penalty_factor * (bpe[idx] - mem_budget[idx])
        idx = bpe < 0
        penalty[idx] = self.penalty_factor * (0 - bpe[idx])

        bpe[bpe > mem_budget] = mem_budget[bpe > mem_budget]
        bpe[bpe < 0] = 0

        return bpe, penalty

    def split_tuner_out(self, tuner_out):
        bpe = tuner_out[:, 0]
        bpe = bpe.view(-1, 1)
        categorical_feats = tuner_out[:, 1:]

        return bpe, categorical_feats

    def forward(self, pred, label):
        assert self.model.training is False
        # For learned cost model loss, the prediction is the DB configuration
        # and label is the workload
        bpe, categorical_feats = self.split_tuner_out(pred)
        bpe, penalty = self.calc_mem_penalty(label, bpe)

        inputs = torch.concat([label, bpe, categorical_feats], dim=-1)
        out = self.model(inputs)
        out = out.sum(dim=-1)
        out = out + penalty
        out = out.mean()

        return out
