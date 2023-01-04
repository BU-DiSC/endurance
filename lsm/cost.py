import lsm.lsm_models as Model
from typing import Self
from lsm.lsmtype import Policy


class EndureTierLevelCost:
    def __init__(self, config: dict[str, ...]) -> Self:
        self._config = config
        self.cf = Model.EndureTierLevelCost(**self._config['system'])

    def Z0(self, h: float, T: float, policy: Policy):
        return self.cf.Z0(h, T, policy)

    def Z1(self, h: float, T: float, policy: Policy):
        return self.cf.Z1(h, T, policy)

    def W(self, h: float, T: float, policy: Policy):
        return self.cf.W(h, T, policy)

    def Q(self, h: float, T: float, policy: Policy):
        return self.cf.Q(h, T, policy)

    def calc_cost(
        self,
        h: float,
        T: float,
        policy: Policy,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        return ((z0 * self.Z0(h, T, policy))
                + (z1 * self.Z1(h, T, policy))
                + (q * self.Q(h, T, policy))
                + (w * self.W(h, T, policy)))

    def __call__(
        self,
        h: float,
        T: float,
        policy: Policy,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        self.calc_cost(h, T, policy, z0, z1, q, w)
