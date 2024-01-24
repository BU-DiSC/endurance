from endure.lsm.types import Policy

kSYSTEM_HEADER = [
    "entry_p_page",
    "selec",
    "entry_size",
    "max_h",
    "num_elem"
]

kWORKLOAD_HEADER = [
    "z0",
    "z1",
    "q",
    "w",
]

kCOST_HEADER = [
    "z0_cost",
    "z1_cost",
    "q_cost",
    "w_cost",
]

kINPUT_FEATS_DICT = {
    Policy.Tiering: kSYSTEM_HEADER + kWORKLOAD_HEADER + ["h", "T"],
    Policy.Leveling: kSYSTEM_HEADER + kWORKLOAD_HEADER + ["h", "T"],
    Policy.Classic: kSYSTEM_HEADER + kWORKLOAD_HEADER + ["h", "policy", "T"],
    Policy.QFixed: kSYSTEM_HEADER + kWORKLOAD_HEADER + ["h", "T", "Q"],
    Policy.YZHybrid: kSYSTEM_HEADER + kWORKLOAD_HEADER + ["h", "T", "Y", "Z"],
    Policy.KHybrid: kSYSTEM_HEADER + kWORKLOAD_HEADER + ["h", "T", "K"],
}
