from typing import Any, Literal

from pydantic import BaseModel, Field

from .lsm.types import Policy


class LSMBounds(BaseModel):
    max_considered_levels: int = 20
    bits_per_elem_range: tuple[float, float] = (1, 10)
    size_ratio_range: tuple[int, int] = (2, 31)
    page_sizes: tuple[int, ...] = (4, 8, 16)
    entry_sizes: tuple[int, ...] = (1024, 2048, 4096, 8192)
    memory_budget_range: tuple[float, float] = (5.0, 20.0)
    selectivity_range: tuple[float, float] = (1e-7, 1e-9)
    elements_range: tuple[int, int] = (100000000, 5000000000)


class LossConfig(BaseModel):
    name: str = Field(default="MSE")  # could be a str path for ModelBasedLoss
    is_lcm: bool = Field(default=False)  # will indicate if it is an LCM
    options: dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    name: Literal["Adam", "AdamW", "SGD", "Adagrad"] = Field(default="Adam")
    options: dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(BaseModel):
    name: Literal["CosineAnnealingLR", "Exponential", "Constant"] = Field(
        default="Constant"
    )
    options: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    max_epochs: int = 10
    data_split: float = 0.9
    batch_size: int = 64
    shuffle: bool = False
    num_workers: int = 1
    checkpoints: bool = False
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)


class LCMConfig(BaseModel):
    hidden_length: int = Field(default=1, ge=1)
    hidden_width: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    embedding_size: int = Field(default=8, ge=1)
    decision_dim: int = Field(default=64, ge=10)
    norm_layer: Literal["Batch", "Norm"] = "Batch"
    # Used only for classic models, generally smaller than embedding size
    policy_embedding_size: int = Field(default=4)


class LTunerConfig(BaseModel):
    penalty_factor: int = Field(default=10, ge=1)
    hidden_length: int = Field(default=1, ge=1)
    hidden_width: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    norm_layer: Literal["Batch", "Norm"] = "Batch"
    categorical_mode: Literal["gumbel", "reinmax"] = "gumbel"
    train_kwargs: dict[str, Any] = Field(default={"temp": 10, "hard": False})
    validate_kwargs: dict[str, Any] = Field(default={"temp": 0.001, "hard": False})


class LSMConfig(BaseModel):
    policy: Policy = Field(default=Policy.Classic)
    bounds: LSMBounds = Field(default_factory=LSMBounds)


class AxeConfig(BaseModel):
    use_gpu: bool = Field(default=False)
    disable_tqdm: bool = Field(default=False)
    seed: int = Field(default=2169)
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = Field(
        default="INFO"
    )
    lsm: LSMConfig = Field(default_factory=LSMConfig)
    ltuner: LTunerConfig = Field(default_factory=LTunerConfig)
    lcm: LCMConfig = Field(default_factory=LCMConfig)
    lcm_train: TrainerConfig = Field(default_factory=TrainerConfig)
    ltuner_train: TrainerConfig = Field(default_factory=TrainerConfig)
