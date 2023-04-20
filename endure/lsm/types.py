import typing
import enum


class Policy(enum.Enum):
    Tiering = 0
    Leveling = 1
    KHybrid = 2
    QFixed = 3
    YZHybrid = 4


class LSMSystem(typing.NamedTuple):
    B: int
    E: int
    H: int
    N: int
    phi: float
    s: float


class LSMTree:
    MAX_LEVELS = 12

    def __init__(
        self,
        system: LSMSystem,
        h: float = 5.0,
        T: float = 5.0,
        policy: Policy = Policy.Tiering,
        Q: float = 1.0,
        K: list[float] = None,
        Y: float = 1.0,
        Z: float = 1.0,
    ) -> None:
        self.system = system
        self.h = h
        self.T = T
        self.policy = policy
        if K is None:
            self.K = [1.0] * self.MAX_LEVELS
        else:
            self.K = K
        self.Q = Q
        self.Y = Y
        self.Z = Z

    def as_dict(self) -> dict:
        d = {
            "B": self.system.B,
            "E": self.system.E,
            "H": self.system.H,
            "N": self.system.N,
            "phi": self.system.phi,
            "s": self.system.s,
            "T": self.T,
            "h": self.h,
            "policy": self.policy,
            "Q": self.Q,
            "K": self.K,
            "Y": self.Y,
            "Z": self.Z,
        }

        return d
