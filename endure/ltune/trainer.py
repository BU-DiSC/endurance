import torch
from typing import Optional, Union
from torch.utils.data import DataLoader, Dataset


class LTuneTrainer:
    def __init__(
        self,
        config: dict[str, ...],
        model: torch.nn.Modue,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_data: Union[DataLoader, Dataset],
        test_data: Union[DataLoader, Dataset],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self._config = config
