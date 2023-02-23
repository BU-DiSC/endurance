import os
import torch
from endure.lcm.model.builder import LearnedCostModelBuilder


class LearnedCostModelLoss(torch.nn.Module):
    def __init__(self, config: dict[str, ...]):
        super().__init__()
        self.lcm_builder = LearnedCostModelBuilder(config)
        self.model = self.lcm_builder.build_model()
        _, extension = os.path.splitext(config['ltune']['loss_model_path'])
        is_checkpoint = extension == '.checkpoint'
        data = torch.load(os.path.join(
            config['io']['data_dir'],
            config['ltune']['loss_model_path']))
        if is_checkpoint:
            data = data['model_state_dict']
        status = self.model.load_state_dict(data)
        assert (status.missing_keys == 0
                and status.unexpected_keys == 0)

    def forward(self, pred, label):
        # For learned cost model loss, pred is the DB configuration, label is
        # the workload
        bpe = pred[:, 0].view(-1, 1)
        size_ratio = torch.argmax(pred[:, 1:], dim=-1).view(-1, 1)
        inputs = torch.concat([label, bpe, size_ratio])

        return self.model(inputs).sum()
