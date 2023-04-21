import os
import torch
from typing import Any

from torch.nn.parameter import Parameter

from endure.lcm.model.builder import LearnedCostModelBuilder


class LearnedCostModelLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any], model_path: str):
        super().__init__()
        bpe_max = config["lsm"]["bits_per_elem"]["max"]
        bpe_min = config["lsm"]["bits_per_elem"]["min"]
        self._bpe_mean = Parameter(torch.Tensor([(bpe_max + bpe_min) / 2]))
        self._bpe_std = Parameter(
            torch.sqrt(torch.Tensor([(bpe_max - bpe_min) ** 2 / 12]))
        )
        self.normalize_bpe = config["ltune"]["data"]["normalize"]

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

    def forward(self, pred, label):
        # For learned cost model loss, pred is the DB configuration, label is
        # the workload

        bpe = pred[:, 0]
        size_ratio = torch.argmax(pred[:, 1:], dim=-1).view(-1, 1)

        if self.normalize_bpe:
            bpe = ((bpe - self._bpe_mean) / self._bpe_std).view(-1, 1)

        inputs = torch.concat([label, bpe, size_ratio], dim=-1)
        out = self.model(inputs)

        return out.sum(dim=-1).square().mean()
