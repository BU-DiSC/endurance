import os
import logging
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pa
import torch
from torch.utils.data import Dataset


class LCMDataSet(Dataset):
    def __init__(self, config, folder, format="csv"):
        self._config: dict = config
        self.log: logging.Logger = logging.getLogger(config["log"]["name"])
        self._format: str = format

        fnames = glob.glob(os.path.join(folder, "*." + self._format))
        self.log.info("Loading in all files to RAM")
        self._df = self._load_data(fnames)

        self.inputs = torch.from_numpy(self._df[self._get_input_cols()].values).float()
        self.labels = torch.from_numpy(self._df[self._get_output_cols()].values).float()

    def _get_output_cols(self):
        return self._config["lcm"]["output_features"]

    def _get_input_cols(self) -> list[str]:
        base_features: list[str] = self._config["lcm"]["input_features"]
        if "K" in base_features:
            k_cols = [f"K_{i}" for i in range(self._config["lsm"]["max_levels"])]
            base_features = list(filter(lambda x: x != "K", base_features))
            base_features = base_features + k_cols

        return base_features

    def _load_data(self, fnames):
        tables = []
        for fname in fnames:
            tables.append(pa.read_table(fname).to_pandas())
        tables = pd.concat(tables)

        tables = self._sanitize_tables(tables)
        if self._config["lcm"]["data"]["normalize_inputs"]:
            tables = self._normalize_tables(tables)

        return tables

    def _normalize_tables(self, df: pd.DataFrame) -> pd.DataFrame:
        df[["z0", "z1", "q", "w"]] -= np.array([0.5, 0.5, 0.5, 0.5])
        df[["z0", "z1", "q", "w"]] /= np.array([0.3, 0.3, 0.3, 0.3])
        df[["B", "s", "E", "H", "N"]] -= df[["B", "s", "E", "H", "N"]].mean()
        df[["B", "s", "E", "H", "N"]] /= df[["B", "s", "E", "H", "N"]].std()

        return df

    def _sanitize_tables(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        return self.labels[idx], self.inputs[idx]
