import pytest
import torch

from endure.ltune.model.qlsm_tuner import QLSMTuner


@pytest.mark.parametrize("num_feats", [3])
@pytest.mark.parametrize("capacity_range", [8])
def test_qlsm_tuner_train_shape(num_feats, capacity_range):
    batch = 4
    output_len = 1 + (2 * capacity_range)

    tuner = QLSMTuner(num_feats=num_feats, capacity_range=capacity_range)
    tuner.train()
    x = torch.ones(batch, num_feats)
    x = x.view(batch, -1)
    out = tuner(x)

    assert out.shape == torch.Size([batch, output_len])
    assert out.sum().item() != float("nan")
    assert out.sum().item() != float("inf")


@pytest.mark.parametrize("num_feats", [3])
@pytest.mark.parametrize("capacity_range", [8])
def test_qlsm_tuner_test_shape(num_feats, capacity_range):
    # bits, T (one_hot), Q (one_hot)
    output_len = 1 + (2 * capacity_range)

    tuner = QLSMTuner(num_feats=num_feats, capacity_range=capacity_range)
    tuner.eval()
    x = torch.ones(1, num_feats)
    x = x.view(1, -1)
    out = tuner(x)

    assert out.shape == torch.Size([1, output_len])
    assert out.sum().item() != float("nan")
    assert out.sum().item() != float("inf")
