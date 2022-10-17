#!/usr/bin/env python

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations_with_replacement
from data.io import Reader
import lsm.cost as CostFunc
import multiprocessing as mp

NUM_FILES = 8
TMAX = 50
TLOW = 2
MAX_LEVELS = 16
HMAX = 9.5
HLOW = 0
WL_DIM = 4
PRECISION = 3
SAMPLES = 2**16

config = Reader.read_config("config/endure.toml")
reader = Reader(config)
cf = CostFunc.EndureKHybridCost(**config["system"])


def gen_workload(dim: int, max_val: float) -> list:
    # See the stackoverflow thread for why the simple solution is not uniform
    # https://stackoverflow.com/questions/8064629/random-numbers-that-add-to-100-matlab
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
        "z0_cost": z0_cost,
        "z1_cost": z1_cost,
        "q_cost": q_cost,
        "w_cost": w_cost,
        "h": h,
        "z0": z0,
        "z1": z1,
        "q": q,
        "w": w,
        "T": T,
    }
    for level_idx in range(MAX_LEVELS):
        row[f"K_{level_idx}"] = K[level_idx]

    return row


def gen_file(idx: int) -> int:
    output_dir = os.path.join(
        config['io']['data_dir'],
        config['io']['train_dir_name'])
    fname = f'train_{idx:03}.csv'

    df = []
    pos = mp.current_process()._identity[0] - 1
    for _ in tqdm(range(SAMPLES), desc=fname, position=pos, ncols=80):
        z0, z1, q, w = gen_workload(WL_DIM, 1)
        T = np.random.randint(low=TLOW, high=TMAX)
        h = np.around(HMAX * np.random.rand(), PRECISION)

        levels = int(cf.L(h, T, True))
        K = random.sample(create_k_levels(levels, T - 1), 1)[0]
        K = np.pad(K, (0, MAX_LEVELS - len(K)))
        row = create_row(h, T, K, z0, z1, q, w)
        df.append(row)

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(output_dir, fname))

    return idx


def gen_files() -> None:
    output_dir = os.path.join(
        config["io"]["data_dir"],
        config["io"]["train_dir_name"])
    print(f'Writing all files to {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    inputs = list(range(0, NUM_FILES))
    with mp.Pool(
        mp.cpu_count(),
        initializer=tqdm.set_lock,
        initargs=(mp.RLock(), )
    ) as p:
        p.map(gen_file, inputs)

    return


if __name__ == "__main__":
    gen_files()
