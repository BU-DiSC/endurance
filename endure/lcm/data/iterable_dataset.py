import glob
import logging
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pa
import torch
import torch.utils.data


class LCMIterableDataSet(torch.utils.data.IterableDataset):
    def __init__(self, config, folder, format="csv", shuffle=False):
        self._config = config
        self.log = logging.getLogger(config["log"]["name"])
        self._mean = np.array(config["lcm"]["data"]["mean_bias"], np.float32)
        self._std = np.array(config["lcm"]["data"]["std_bias"], np.float32)
        self._label_cols = ["z0_cost", "z1_cost", "q_cost", "w_cost"]
        self._input_cols = self._get_input_cols()
        self._format = format
        self._fnames = glob.glob(os.path.join(folder, "*." + format))
        self._shuffle = shuffle

    def _get_input_cols(self) -> list[str]:
        base = ["z0", "z1", "q", "w", "B", "s", "E", "H", "N", "h", "T"]
        choices = {
            "KLSM": [f"K_{i}" for i in range(self._config["lsm"]["max_levels"])],
            "Tier": [],
            "Level": [],
            "QLSM": ["Q"],
        }
        extension = choices.get(self._config["lsm"]["design"], None)
        if extension is not None:
            input_cols = base + extension
        else:
            self.log.warn("Invalid design defaulting to Level")
            input_cols = base

        return input_cols

    def _load_data(self, fname) -> pd.DataFrame:
        if self._format == "parquet":
            df = pa.read_table(fname).to_pandas()
        else:  # default csv
            df = pd.read_csv(fname)

        df = self._sanitize_df(df)
        if self._config["lcm"]["data"]["normalize_inputs"]:
            df = self._normalize_df(df)

        return df

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df[["z0", "z1", "q", "w", "h"]] -= self._mean
        df[["z0", "z1", "q", "w", "h"]] /= self._std
        df[["B", "s", "E", "H", "N"]] -= df[["B", "s", "E", "H", "N"]].mean()

        return df

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["T"] = df["T"] - self._config["lsm"]["size_ratio"]["min"]
        if self._config["lsm"]["design"] == "QLSM":
            df["Q"] -= self._config["lsm"]["size_ratio"]["min"] - 1
        elif self._config["lsm"]["design"] == "KLSM":
            for i in range(self._config["lsm"]["max_levels"]):
                df[f"K_{i}"] -= self._config["lsm"]["size_ratio"]["min"] - 1
                df[f"K_{i}"][df[f"K_{i}"] < 0] = 0
        elif self._config["lsm"]["design"] in ["Tier", "Level"]:
            pass
        else:
            self.log.warn("Invalid model defaulting to KCost behavior")
            for i in range(self._config["lsm"]["max_levels"]):
                df[f"K_{i}"] -= self._config["lsm"]["size_ratio"]["min"] - 1
                df[f"K_{i}"][df[f"K_{i}"] < 0] = 0

        return df

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self._fnames
        else:
            if self._shuffle:
                np.random.shuffle(self._fnames)
            file_bins = np.array_split(self._fnames, worker_info.num_workers)
            files = file_bins[worker_info.id]
        if self._shuffle:
            np.random.shuffle(files)
        for file in files:
            df = self._load_data(file)
            labels = torch.from_numpy(df[self._label_cols].values).float()
            inputs = torch.from_numpy(df[self._input_cols].values).float()
            indices = list(range(len(labels)))
            if self._shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                yield labels[idx], inputs[idx]
            del df  # attempt to release dataframe memory
