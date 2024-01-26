from dataclasses import dataclass, field
import enum


class Policy(enum.Enum):
    Tiering = 0
    Leveling = 1
    Classic = 2
    KHybrid = 3
    QFixed = 4
    YZHybrid = 5


STR_POLICY_DICT = {
    "Tier": Policy.Tiering,
    "Level": Policy.Leveling,
    "Classic": Policy.Classic,
    "KHybrid": Policy.KHybrid,
    "QFixed": Policy.QFixed,
    "YZHybrid": Policy.YZHybrid,
}


@dataclass
class System:
    E: int = 8192  # Number of physical entries per page
    s: float = 4e-7  # Range query selectivity
    B: int = 4  # entries per page
    N: int = 1_000_000_000  # Number of elements in tree
    H: float = 10.0  # Bits per element budget
    phi: float = 1.0  # Read/Write asymmetry coefficient


@dataclass
class LSMDesign:
    h: float = 5.0
    T: float = 5.0
    policy: Policy = Policy.Leveling
    Q: float = 1.0
    Y: float = 1.0
    Z: float = 1.0
    K: list[float] = field(default_factory=list)


@dataclass
class Workload:
    z0: float = 0.25
    z1: float = 0.25
    q: float = 0.25
    w: float = 0.25
