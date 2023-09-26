import os
import glob
import random
import torch
import numpy as np
import pyarrow.dataset as ds


class ParquetBatchDataSet:
    def __init__(
        self,
        config: dict,
        path: str,
        batch_size: int,
        use_threads: bool = True,
        count_rows: bool = False,
        shuffle: bool = True,
    ):
        self._config = config
        self._path = path
        self._batch_size = batch_size
        self._use_threads = use_threads
        self._count_rows = count_rows
        self._shuffle = shuffle

        self._mean = np.array(self._config["lcm"]["data"]["mean_bias"], np.float32)
        self._std = np.array(self._config["lcm"]["data"]["std_bias"], np.float32)
        self._label_cols = ["z0_cost", "z1_cost", "q_cost", "w_cost"]
        self._input_cols = self._get_input_cols()

        self._files = glob.glob(os.path.join(self._path, "*.parquet"))
        self._dataset = ds.dataset(self._files)
        self._scan = self._dataset.scanner(
            batch_size=self._batch_size, use_threads=self._use_threads
        )
        self._batches = self._scan.to_batches()
        self._nrows = self._scan.count_row() if self._count_rows else None

    def _get_input_cols(self):
        base = ["h", "z0", "z1", "q", "w", "T"]
        choices = {
            "KCost": [f"K_{i}" for i in range(self._config["lsm"]["max_levels"])],
            "TierLevelCost": [],
            "LevelCost": [],
            "TierCost": [],
            "QCost": ["Q"],
        }
        extension = choices.get(self._config["lsm"]["design"], None)
        if extension is None:
            self.log.warn("Invalid model defaulting to KCost")
            extension = choices.get("KCost")

        return base + extension

    def _process_df(self, df):
        df[["h", "z0", "z1", "q", "w"]] -= self._mean
        df[["h", "z0", "z1", "q", "w"]] /= self._std
        df["T"] = df["T"] - self._config["lsm"]["size_ratio"]["min"]
        if self._config["lsm"]["design"] in ["QLSM", "QLSMIntegerVars"]:
            df["Q"] -= self._config["lsm"]["size_ratio"]["min"] - 1
        elif self._config["lsm"]["design"] == "TierLevelCost":
            pass
        elif self._config["lsm"]["design"] == "KCost":
            for i in range(self._config["lsm"]["max_levels"]):
                df[f"K_{i}"] -= self._config["lsm"]["size_ratio"]["min"] - 1
                df[f"K_{i}"][df[f"K_{i}"] < 0] = 0
        else:
            self.log.warn("Invalid model defaulting to KCost behavior")
            for i in range(self._config["lsm"]["max_levels"]):
                df[f"K_{i}"] -= self._config["lsm"]["size_ratio"]["min"] - 1
                df[f"K_{i}"][df[f"K_{i}"] < 0] = 0

        return df

    def reset(self):
        self._batches = self._scan.to_batches()
        if self._shuffle:
            random.shuffle(self._files)
        return

    def __iter__(self):
        return self

    def __next__(self):
        df = next(self._batches).to_pandas()
        df = self._process_df(df)
        return (
            torch.from_numpy(df[self._label_cols].values).long(),
            torch.from_numpy(df[self._input_cols].values).long(),
        )
