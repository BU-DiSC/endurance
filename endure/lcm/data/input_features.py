from endure.lsm.types import Policy

_kBASE = ["z0", "z1", "q", "w", "B", "s", "E", "max_h", "N"]

kINPUT_FEATS_DICT = {
    Policy.Tiering: _kBASE + ["h", "T"],
    Policy.Leveling: _kBASE + ["h", "T"],
    Policy.Classic: _kBASE + ["h", "policy", "T"],
    Policy.QFixed: _kBASE + ["h", "T", "Q"],
    Policy.YZHybrid: _kBASE + ["h", "T", "Y", "Z"],
    Policy.KHybrid: _kBASE + ["h", "T", "K"],
}
