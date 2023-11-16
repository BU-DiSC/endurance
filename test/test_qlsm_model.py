import pytest
import torch

from endure.lcm.util import one_hot_lcm
from endure.lcm.model import QModel, QModelMultiHead


@pytest.mark.parametrize("num_feats", [10])
@pytest.mark.parametrize("capacity_range", [20])
@pytest.mark.parametrize("out_width", [4])
@pytest.mark.parametrize("model_type", [QModelMultiHead, QModel])
def test_qlsm_lcm_train_shape(num_feats, capacity_range, out_width, model_type):
    model = model_type(
        num_feats=num_feats,
        capacity_range=capacity_range,
        out_width=out_width,
        hidden_length=1,
    )
    model.train()
    x = torch.ones(2, num_feats)
    x = x.view(2, -1)
    print(x.shape)
    out = model(x)
    assert out.shape == torch.Size([2, out_width])
    assert out.sum().item() != float('nan')
    assert out.sum().item() != float('inf')


@pytest.mark.parametrize("num_feats", [10])
@pytest.mark.parametrize("capacity_range", [20])
@pytest.mark.parametrize("out_width", [4])
@pytest.mark.parametrize("model_type", [QModelMultiHead, QModel])
def test_qlsm_lcm_eval_shape(num_feats, capacity_range, out_width, model_type):
    model = model_type(
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


@pytest.mark.parametrize("num_feats", [10])
@pytest.mark.parametrize("capacity_range", [20])
@pytest.mark.parametrize("out_width", [4])
@pytest.mark.parametrize("model_type", [QModelMultiHead, QModel])
def test_qlsm_lcm_eval_deterministic(
    num_feats,
    capacity_range,
    out_width,
    model_type
):
    model = model_type(
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
