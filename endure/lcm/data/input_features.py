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
    Policy.Tiering: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h", "T"],
    Policy.Leveling: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h", "T"],
    Policy.Classic: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h", "policy", "T"],
    Policy.QFixed: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h", "T", "Q"],
    Policy.YZHybrid: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h", "T", "Y", "Z"],
    Policy.KHybrid: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h", "T", "K"],
}

kOUTPUT_FEATS = ["z0_cost", "z1_cost", "q_cost", "w_cost"]
