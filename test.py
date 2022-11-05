#!/usr/bin/env python
from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dp = IterableWrapper(range(5)).shuffle().sharding_filter()
    print(list(DataLoader(dp, num_workers=2, shuffle=True)))
