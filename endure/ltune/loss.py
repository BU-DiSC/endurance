import os
import torch
from endure.lcm.model.builder import LearnedCostModelBuilder


class LearnedCostModelLoss(torch.nn.Module):
    def __init__(self, config: dict[str, ...], model_path: str):
        super().__init__()
        self.lcm_builder = LearnedCostModelBuilder(config)
        self.model = self.lcm_builder.build_model()
        _, extension = os.path.splitext(model_path)
        is_checkpoint = (extension == '.checkpoint')
        data = torch.load(os.path.join(config['io']['data_dir'], model_path))
        if is_checkpoint:
            data = data['model_state_dict']
        status = self.model.load_state_dict(data)
        assert (len(status.missing_keys) == 0)
        assert (len(status.unexpected_keys) == 0)

    def forward(self, pred, label):
        # For learned cost model loss, pred is the DB configuration, label is
        # the workload

        # TODO normalize BPE by calculating mean and std from max/min
        bpe = ((pred[:, 0] - 5) / 2.88).view(-1, 1)
        size_ratio = torch.argmax(pred[:, 1:], dim=-1).view(-1, 1)
        inputs = torch.concat([bpe, label, size_ratio], dim=-1)

        return self.model(inputs).sum(dim=-1).mean()
