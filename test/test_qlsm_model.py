import pytest
import torch

from endure.lcm.util import one_hot_lcm
from endure.lcm.model import QModel


@pytest.mark.parametrize("num_feats", [10, 4])
@pytest.mark.parametrize("capacity_range", [10, 50])
@pytest.mark.parametrize("out_width", [1, 4])
def test_qlsm_training_mode_shape(num_feats, capacity_range, out_width):
    model = QModel(
        num_feats=num_feats,
        capacity_range=capacity_range,
        out_width=out_width,
        hidden_length=1,
    )
    model.train()
    x = torch.ones(num_feats)
    x = x.view(1, -1)
    out = model(x)
    assert out.shape == torch.Size([1, out_width])
    assert out.sum().item() != float('nan')
    assert out.sum().item() != float('inf')


@pytest.mark.parametrize("num_feats", [4, 15])
@pytest.mark.parametrize("capacity_range", [4, 10])
@pytest.mark.parametrize("out_width", [1, 8])
def test_qlsm_evaluation_mode_shape(num_feats, capacity_range, out_width):
    model = QModel(
        num_feats=num_feats,
        capacity_range=capacity_range,
        out_width=out_width,
        hidden_length=1,
    )
    model.eval()
    x = torch.ones(num_feats)
    x = one_hot_lcm(x, num_feats, 2, capacity_range)
    x = x.view(1, -1)
    out = model(x)
    assert out.shape == torch.Size([1, out_width])
    assert out.sum().item() != float('nan')
    assert out.sum().item() != float('inf')


@pytest.mark.parametrize("num_feats", [8, 9])
@pytest.mark.parametrize("capacity_range", [2, 40])
@pytest.mark.parametrize("out_width", [1, 5])
def test_qlsm_eval_deterministic(num_feats, capacity_range, out_width):
    model = QModel(
        num_feats=num_feats,
        capacity_range=capacity_range,
        out_width=out_width,
        hidden_length=1,
    )
    model.eval()
    x = torch.ones(num_feats)
    x = one_hot_lcm(x, num_feats, 2, capacity_range)
    x = x.view(1, -1)
    out_x = model(x)

    y = torch.ones(num_feats)
    y = one_hot_lcm(y, num_feats, 2, capacity_range)
    y = y.view(1, -1)
    out_y = model(y)

    assert torch.equal(out_x, out_y)
