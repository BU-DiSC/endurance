import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations_with_replacement
from data.io import Reader
import lsm.cost as CostFunc

TMAX = 50
MAX_LEVELS = 16
SAMPLES = 1000000

config = Reader.read_config('config/endure.toml')
reader = Reader(config)


def create_k_levels(levels: int, max_T: int):
    arr = combinations_with_replacement(range(max_T,  0, -1), levels)
    return list(arr)


def gen_file(file_id):
    fname = f'train_{file_id}.feather'
    print(f'Generating file {fname}')
    cf = CostFunc.EndureKHybridCost(**config['system'])
    wls = np.random.rand(SAMPLES, 4)
    wls = np.around(wls / wls.sum(axis=1).reshape(SAMPLES, 1), 3)
    hs = np.around(9.5 * np.random.rand(SAMPLES), 2)
    Ts = np.random.randint(low=2, high=TMAX, size=SAMPLES)

    df = []
    for wl, h, T in tqdm(zip(wls, hs, Ts), total=SAMPLES, ncols=80):
        z0, z1, q, w = wl
        levels = int(cf.L(h, T, True))
        arr = create_k_levels(levels, T - 1)
        arr = random.sample(arr, min(10, len(arr)))
        # tier, level = np.array([T - 1] * levels), np.array([1] * levels)
        for K in arr:
            K = np.pad(K, (0, MAX_LEVELS - len(K)))
            k_cost = cf.calc_cost(h, T, K, z0, z1, q, w)
            row = {
                'h': h,
                'T': T,
                'z0': z0,
                'z1': z1,
                'q': q,
                'w': w,
                'B': config['system']['B'],
                'phi': config['system']['phi'],
                's': config['system']['s'],
                'E': config['system']['E'],
                'H': config['system']['H'],
                'N': config['system']['N'],
                'k_cost': k_cost,
            }
            for level_idx in range(MAX_LEVELS):
                row[f'K_{level_idx}'] = K[level_idx]
            df.append(row)

    df = pd.DataFrame(df)
    df.to_feather('training_data/' + fname)


for idx in range(20):
    gen_file(idx)
