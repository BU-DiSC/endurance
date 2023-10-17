import pytest
import torch

from endure.lcm.model import QModel
from endure.data.io import Reader


@pytest.fixture(autouse=True)
def read_config():
    config = Reader.read_config("endure.toml")
    return config


def test_qlsm_training_mode_shape():
    num_feats, capacity_range, out_width = 13, 10, 4
    model = QModel(
        num_feats=num_feats,
        capacity_range=capacity_range,
        out_width=out_width,
        hidden_length=1,
    )
    model.train()
    x = torch.ones([1, num_feats])
    out = model(x)
    assert out.shape == torch.Size([1, out_width])
    assert out.sum().item() != float('nan')
    assert out.sum().item() != float('inf')
