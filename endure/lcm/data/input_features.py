from endure.lsm.types import Policy

kSYSTEM_HEADER = ["entry_p_page", "selec", "entry_size", "max_h", "num_elem"]

kWORKLOAD_HEADER = [
    "z0_percent",
    "z1_percent",
    "q_percent",
    "w_percent",
]

kCOST_HEADER = [
    "z0_cost",
    "z1_cost",
    "q_cost",
    "w_cost",
]

kINPUT_FEATS_DICT = {
    Policy.Tiering: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h_val", "T_val"],
    Policy.Leveling: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h_val", "T_val"],
    Policy.Classic: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h_val", "policy", "T_val"],
    Policy.QFixed: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h_val", "T_val", "Q_val"],
    Policy.YZHybrid: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h_val", "T_val", "Y_val", "Z_val"],
    Policy.KHybrid: kWORKLOAD_HEADER + kSYSTEM_HEADER + ["h_val", "T_val", "K_val"],
}

kOUTPUT_FEATS = ["z0_cost", "z1_cost", "q_cost", "w_cost"]
