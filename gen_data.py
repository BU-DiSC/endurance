#!/usr/bin/env python

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations_with_replacement
from data.io import Reader
import lsm.cost as CostFunc

NUM_FILES = 100
TMAX = 50
TLOW = 2
MAX_LEVELS = 16
HMAX = 9.5
HLOW = 0
WL_DIM = 4
PRECISION = 3
SAMPLES = 2**20

config = Reader.read_config("config/endure.toml")
reader = Reader(config)
cf = CostFunc.EndureKHybridCost(**config["system"])


# See the stackoverflow thread for why the simple solution is not uniform
# https://stackoverflow.com/questions/8064629/random-numbers-that-add-to-100-matlab
def gen_workload(dim: int, max_val: float) -> list:
    wl = list(np.random.rand(dim - 1)) + [0, 1]
    wl.sort()
    return [b - a for a, b in zip(wl, wl[1:])]


def create_k_levels(levels: int, max_T: int) -> list:
    arr = combinations_with_replacement(range(max_T, 0, -1), levels)
    return list(arr)


def create_row(h, T, K, z0, z1, q, w) -> dict:
    z0_cost = z0 * cf.Z0(h, T, K)
    z1_cost = z1 * cf.Z1(h, T, K)
    q_cost = q * cf.Q(h, T, K)
    w_cost = w * cf.W(h, T, K)
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
        "z0_cost": z0_cost,
        "z1_cost": z1_cost,
        "q_cost": q_cost,
        "w_cost": w_cost,
        "k_cost": z0_cost + z1_cost + q_cost + w_cost,
    }
    for level_idx in range(MAX_LEVELS):
        row[f"K_{level_idx}"] = K[level_idx]

    return row


def generate_random_list() -> list:
    df = []
    for _ in tqdm(range(SAMPLES), ncols=80):
        z0, z1, q, w = gen_workload(WL_DIM, 1)
        T = np.random.randint(low=TLOW, high=TMAX)
        h = np.around(HMAX * np.random.rand(), PRECISION)

        levels = int(cf.L(h, T, True))
        K = random.sample(create_k_levels(levels, T - 1), 1)[0]
        K = np.pad(K, (0, MAX_LEVELS - len(K)))
        row = create_row(h, T, K, z0, z1, q, w)
        df.append(row)

    return df


def gen_files() -> None:
    output_dir = os.path.join(config["io"]["cold_data_dir"], "training_data")
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(NUM_FILES):
        fname = os.path.join(output_dir, f"train_{idx:04}.feather")
        print(f"Generating file: {fname}")

        df = pd.DataFrame(generate_random_list())
        df.to_feather(fname)
        del df


if __name__ == "__main__":
    gen_files()
