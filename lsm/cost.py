import numpy as np
import lsm.lsm_models as Model


class EndureTierCost:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.cf = Model.EndureKHybridCost(**self._config['system'])

    def Z0(self, h: float, T: float):
        k = np.full(self._config['lsm']['max_levels'], T - 1)
        return self.cf.Z0(h, T, k)

    def Z1(self, h: float, T: float):
        k = np.full(self._config['lsm']['max_levels'], T - 1)
        return self.cf.Z1(h, T, k)

    def W(self, h: float, T: float):
        k = np.full(self._config['lsm']['max_levels'], T - 1)
        return self.cf.W(h, T, k)

    def Q(self, h: float, T: float):
        k = np.full(self._config['lsm']['max_levels'], T - 1)
        return self.cf.Q(h, T, k)

    def __call__(
        self,
        h: float,
        T: float,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        return ((z0 * self.Z0(h, T))
                + (z1 * self.Z1(h, T))
                + (q * self.Q(h, T))
                + (w * self.W(h, T)))


class EndureLevelCost:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.cf = Model.EndureKHybridCost(**self._config['system'])

    def Z0(self, h: float, T: float):
        k = np.ones(self._config['lsm']['max_levels'])
        return self.cf.Z0(h, T, k)

    def Z1(self, h: float, T: float):
        k = np.ones(self._config['lsm']['max_levels'])
        return self.cf.Z1(h, T, k)

    def W(self, h: float, T: float):
        k = np.ones(self._config['lsm']['max_levels'])
        return self.cf.W(h, T, k)

    def Q(self, h: float, T: float):
        k = np.ones(self._config['lsm']['max_levels'])
        return self.cf.Q(h, T, k)

    def __call__(
        self,
        h: float,
        T: float,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        return ((z0 * self.Z0(h, T))
                + (z1 * self.Z1(h, T))
                + (q * self.Q(h, T))
                + (w * self.W(h, T)))


class EndureYZCost:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.cf = Model.EndureKHybridCost(**self._config['system'])

    def Z0(self, h: float, T: float, Y: float, Z: float):
        num_levels = int(self.cf.L(h, T, True))
        k = np.full(num_levels - 1, Y)
        k = np.concatenate((k, [Z]))
        k = np.pad(k, (0, self._config['lsm']['max_levels'] - len(k)),
                   'constant', constant_values=(1., 1.))
        return self.cf.Z0(h, T, k)

    def Z1(self, h: float, T: float, Y: float, Z: float):
        num_levels = int(self.cf.L(h, T, True))
        k = np.full(num_levels - 1, Y)
        k = np.concatenate((k, [Z]))
        k = np.pad(k, (0, self._config['lsm']['max_levels'] - len(k)),
                   'constant', constant_values=(1., 1.))
        return self.cf.Z1(h, T, k)

    def W(self, h: float, T: float, Y: float, Z: float):
        num_levels = int(self.cf.L(h, T, True))
        k = np.full(num_levels - 1, Y)
        k = np.concatenate((k, [Z]))
        k = np.pad(k, (0, self._config['lsm']['max_levels'] - len(k)),
                   'constant', constant_values=(1., 1.))
        return self.cf.W(h, T, k)

    def Q(self, h: float, T: float, Y: float, Z: float):
        num_levels = int(self.cf.L(h, T, True))
        k = np.full(num_levels - 1, Y)
        k = np.concatenate((k, [Z]))
        k = np.pad(k, (0, self._config['lsm']['max_levels'] - len(k)),
                   'constant', constant_values=(1., 1.))
        return self.cf.Q(h, T, k)

    def __call__(
        self,
        h: float,
        T: float,
        Y: float,
        Z: float,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        return ((z0 * self.Z0(h, T, Y, Z))
                + (z1 * self.Z1(h, T, Y, Z))
                + (q * self.Q(h, T, Y, Z))
                + (w * self.W(h, T, Y, Z)))


class EndureQCost:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.cf = Model.EndureKHybridCost(**self._config['system'])

    def Z0(self, h: float, T: float, Q: float):
        k = np.full(self._config['lsm']['max_levels'], Q)
        return self.cf.Z0(h, T, k)

    def Z1(self, h: float, T: float, Q: float):
        k = np.full(self._config['lsm']['max_levels'], Q)
        return self.cf.Z1(h, T, k)

    def W(self, h: float, T: float, Q: float):
        k = np.full(self._config['lsm']['max_levels'], Q)
        return self.cf.W(h, T, k)

    def Q(self, h: float, T: float, Q: float):
        k = np.full(self._config['lsm']['max_levels'], Q)
        return self.cf.Q(h, T, k)

    def __call__(
        self,
        h: float,
        T: float,
        Q: float,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        return ((z0 * self.Z0(h, T, Q))
                + (z1 * self.Z1(h, T, Q))
                + (q * self.Q(h, T, Q))
                + (w * self.W(h, T, Q)))


class EndureKCost:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.cf = Model.EndureKHybridCost(**self._config['system'])

    def Z0(self, h: float, T: float, K: np.ndarray):
        return self.cf.Z0(h, T, K)

    def Z1(self, h: float, T: float, K: np.ndarray):
        return self.cf.Z1(h, T, K)

    def W(self, h: float, T: float, K: np.ndarray):
        return self.cf.W(h, T, K)

    def Q(self, h: float, T: float, K: np.ndarray):
        return self.cf.Q(h, T, K)

    def __call__(
        self,
        h: float,
        T: float,
        K: np.ndarray,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        return ((z0 * self.Z0(h, T, K))
                + (z1 * self.Z1(h, T, K))
                + (q * self.Q(h, T, K))
                + (w * self.W(h, T, K)))
