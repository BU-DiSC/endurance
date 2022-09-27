#!/usr/bin/env python

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations_with_replacement
from data.io import Reader
import lsm.cost as CostFunc

NUM_FILES = 30
TMAX = 50
TLOW = 2
MAX_LEVELS = 16
HMAX = 9.5
HLOW = 0
WL_DIM = 4
PRECISION = 3
SAMPLES = 1000000

config = Reader.read_config("config/endure.toml")
reader = Reader(config)
cf = CostFunc.EndureKHybridCost(**config["system"])


def create_k_levels(levels: int, max_T: int):
    arr = combinations_with_replacement(range(max_T, 0, -1), levels)
    return list(arr)


def create_row(h, T, z0, z1, q, w, cost) -> dict:
    row = {
        "h": h,
        "T": T,
        "z0": z0,
        "z1": z1,
        "q": q,
        "w": w,
        "B": config["system"]["B"],
        "phi": config["system"]["phi"],
        "s": config["system"]["s"],
        "E": config["system"]["E"],
        "H": config["system"]["H"],
        "N": config["system"]["N"],
        "k_cost": cost,
    }

    return row


def generate_random_list() -> list:
    df = []
    for _ in tqdm(range(SAMPLES), ncols=80):
        wl = np.random.rand(WL_DIM)
        wl = np.around(wl / wl.sum())
        z0, z1, q, w = wl
        T = np.random.randint(low=TLOW, high=TMAX)
        h = np.around(HMAX * np.random.rand(), PRECISION)

        levels = int(cf.L(h, T, True))
        arr = create_k_levels(levels, T - 1)
        arr = random.sample(arr, min(10, len(arr)))
        for K in arr:
            K = np.pad(K, (0, MAX_LEVELS - len(K)))
            k_cost = cf.calc_cost(h, T, K, z0, z1, q, w)
            row = create_row(h, T, z0, z1, q, w, k_cost)
            for level_idx in range(MAX_LEVELS):
                row[f"K_{level_idx}"] = K[level_idx]
            df.append(row)

    return df


def gen_files():
    output_dir = os.path.join(config["io"]["cold_data_dir"], "training_data")
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(NUM_FILES):
        fname = os.path.join(output_dir, f"train_{idx}.feather")
        print(f"Generating file: {fname}")

        df = pd.DataFrame(generate_random_list())
        df.to_feather(fname)
        del df


if __name__ == "__main__":
    gen_files()
