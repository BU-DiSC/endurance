from typing import Literal
from pydantic import BaseModel, Field
from .lsm.types import LSMBounds, Policy, System



class LTunerModelConfig(BaseModel):
    hidden_length: int = Field(default=1, ge=1)
    hidden_width: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    norm_layer: Literal["Batch", "Norm"] = "Batch"
    categorical_mode: Literal["gumbel", "reinmax"] = "gumbel"


class LSMConfig(BaseModel):
    policy: Policy = Field(default=Policy.Classic)
    bounds: LSMBounds = Field(default_factory=LSMBounds)

class AxeConfig(BaseModel):
    lsm: LSMConfig = Field(default_factory=LSMConfig)
    system: System = Field(default_factory=System)
    use_gpu: bool = Field(default=False)
    show_tqdm: bool = Field(default=False)
