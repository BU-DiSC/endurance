from torch import Tensor
import torch
import torch.nn.functional as F

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
