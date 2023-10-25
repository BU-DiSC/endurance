from typing import Any
from torch import Tensor
import torch
import torch.nn.functional as F

from endure.lsm.types import LSMDesign, Policy, System

def one_hot_lcm(
    data: Tensor,
    num_features: int,
    categorical_features: int,
    categories: int
) -> Tensor:
    capacities = data[num_features - categorical_features :]
    capacities = capacities.to(torch.long)
    capacities = F.one_hot(capacities, num_classes=categories)
    capacities = torch.flatten(capacities)
    out = [data[:num_features - categorical_features], capacities]
    out = torch.cat(out)

    return out

def one_hot_lcm_classic(
    data: Tensor,
    categories: int
) -> Tensor:
    policy = data[-2].to(torch.long)
    policy = F.one_hot(policy, num_classes=2)
    size_ratio = data[-1].to(torch.long)
    size_ratio = F.one_hot(size_ratio, num_classes=categories)
    out = [data[:-2], size_ratio, policy]
    out = torch.cat(out)

    return out

def create_input_from_types(
    design: LSMDesign,
    system: System,
    z0: float,
    z1: float,
    q: float,
    w: float,
    min_t: int,
    max_t: int,
) -> Tensor:
    categories = max_t - min_t + 1
    wl = [z0, z1, q, w]
    sys = [system.B, system.s, system.E, system.H, system.N]
    size_ratio = design.T - min_t
    if design.policy in (Policy.Tiering, Policy.Leveling):
        inputs = wl + sys + [design.h, size_ratio, design.policy.value]
        data = torch.Tensor(inputs)
        out = one_hot_lcm_classic(data, categories)
    else: # design.policy == Policy.QFixed
        inputs = wl + sys + [design.h, size_ratio, design.Q - 1]
        data = torch.Tensor(inputs)
        out = one_hot_lcm(data, len(inputs), 2, categories)

    return out

def eval_lcm_impl(
    design: LSMDesign,
    system: System,
    z0: float,
    z1: float,
    q: float,
    w: float,
    model: torch.nn.Module,
    min_t: int,
    max_t: int,
) -> float:
    x = create_input_from_types(design, system, z0, z1, q, w, min_t, max_t)
    x = x.to(torch.float).view(1, -1)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = pred.sum().item()

    return pred
