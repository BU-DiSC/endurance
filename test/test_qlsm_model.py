import pytest
import torch

from axe.lcm.util import one_hot_lcm
from axe.lcm.model import QModel

OUT_WIDTH = 4


@pytest.mark.parametrize("num_feats", [10])
@pytest.mark.parametrize("capacity_range", [20])
def test_qlsm_lcm_train_shape(num_feats, capacity_range):
    model = QModel(
        num_feats=num_feats,
        capacity_range=capacity_range,
        hidden_length=1,
    )
    model.train()
    x = torch.ones(2, num_feats)
    x = x.view(2, -1)
    print(x.shape)
    out = model(x)
    assert out.shape == torch.Size([2, OUT_WIDTH])
    assert out.sum().item() != float("nan")
    assert out.sum().item() != float("inf")


@pytest.mark.parametrize("num_feats", [10])
@pytest.mark.parametrize("capacity_range", [20])
def test_qlsm_lcm_eval_shape(num_feats, capacity_range):
    model = QModel(
        num_feats=num_feats,
        capacity_range=capacity_range,
        hidden_length=1,
    )
    model.eval()
    x = torch.ones(num_feats)
    x = one_hot_lcm(x, num_feats, 2, capacity_range)
    x = x.view(1, -1)
    out = model(x)
    assert out.shape == torch.Size([1, OUT_WIDTH])
    assert out.sum().item() != float("nan")
    assert out.sum().item() != float("inf")


@pytest.mark.parametrize("num_feats", [10])
@pytest.mark.parametrize("capacity_range", [20])
def test_qlsm_lcm_eval_deterministic(
    num_feats,
    capacity_range,
):
    model = QModel(
        num_feats=num_feats,
        capacity_range=capacity_range,
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
