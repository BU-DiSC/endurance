import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pa
import torch
from torch import Tensor
import torch.utils.data

from endure.lcm.data.input_features import kINPUT_FEATS_DICT
from endure.lsm.types import STR_POLICY_DICT, Policy
from endure.lcm.util import one_hot_lcm, one_hot_lcm_classic


class LCMDataSet(torch.utils.data.IterableDataset):
    kOUTPUT_FEATS = ["z0_cost", "z1_cost", "q_cost", "w_cost"]
    tmp = STR_POLICY_DICT

    def __init__(
        self,
        folder,
        lsm_design: Policy,
        max_levels: int,
        max_size_ratio: int,
        min_size_ratio: int = 2,
        test: bool = False,
        shuffle: bool = False,
    ) -> None:
        self._fnames: list[str] = glob.glob(os.path.join(folder, "*.parquet"))
        self._shuffle: bool = shuffle
        self.lsm_design = lsm_design
        self.max_levels = max_levels
        self.max_size_ratio = max_size_ratio
        self.min_size_ratio = min_size_ratio
        self.categories = max_size_ratio - min_size_ratio + 1
        # When in testing mode we transform input features to one hot encoded
        self.test_mode = test

    def _get_output_cols(self):
        return self.kOUTPUT_FEATS

    def _get_input_cols(self) -> list[str]:
        feats: list[str] = kINPUT_FEATS_DICT[self.lsm_design]
        if "K" in feats:
            k_cols = [f"K_{i}" for i in range(self.max_levels)]
            feats = list(filter(lambda x: x != "K", feats))
            feats = feats + k_cols

        return feats

    def _load_data(self, fname) -> pd.DataFrame:
        df = pa.read_table(fname).to_pandas()
        df = self._sanitize_df(df)

        return df

    def _transform_test_data(self, data: Tensor) -> Tensor:
        num_feat = len(self._get_input_cols())
        if self.lsm_design == Policy.Classic:
            return one_hot_lcm_classic(data, self.categories)
        elif self.lsm_design == Policy.QFixed:
            return one_hot_lcm(data, num_feat, 2, self.categories)
        elif self.lsm_design == Policy.KHybrid:
            return one_hot_lcm(data, num_feat, self.max_levels + 1, self.categories)
        elif self.lsm_design == Policy.YZHybrid:
            raise NotImplementedError
        elif self.lsm_design in [Policy.Leveling, Policy.Tiering]:
            raise NotImplementedError
        else:
            raise TypeError("Incompatible LSM design")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["T"] = df["T"] - self.min_size_ratio
        if self.lsm_design == Policy.QFixed:
            df["Q"] -= self.min_size_ratio - 1
        elif self.lsm_design == Policy.YZHybrid:
            df["Y"] -= self.min_size_ratio - 1
            df["Z"] -= self.min_size_ratio - 1
        elif self.lsm_design == Policy.KHybrid:
            for i in range(self.max_levels):
                df[f"K_{i}"] -= self.min_size_ratio - 1
                df[f"K_{i}"] = df[f"K_{i}"].clip(lower=0)
        elif self.lsm_design in (Policy.Leveling, Policy.Tiering, Policy.Classic):
            pass

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
            labels = torch.from_numpy(df[self._get_output_cols()].values).float()
            inputs = torch.from_numpy(df[self._get_input_cols()].values).float()
            indices = list(range(len(labels)))
            if self._shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                label, input = labels[idx], inputs[idx]
                if self.test_mode:
                    input = self._transform_test_data(inputs[idx])
                yield label, input
