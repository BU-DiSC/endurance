import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pa
import torch
from torch import Tensor
import torch.utils.data

from endure.lcm.data.input_features import kINPUT_FEATS_DICT, kOUTPUT_FEATS
from endure.lsm.types import LSMBounds, Policy
from endure.lcm.util import one_hot_lcm, one_hot_lcm_classic


class LCMDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self,
        folder: str,
        lsm_design: Policy,
        bounds: LSMBounds,
        test: bool = False,
        shuffle: bool = False,
    ) -> None:
        self._fnames: list[str] = glob.glob(os.path.join(folder, "*.parquet"))
        self._shuffle: bool = shuffle
        self.max_levels = bounds.max_considered_levels
        self.min_size_ratio, self.max_size_ratio = bounds.size_ratio_range
        self.categories = self.max_size_ratio - self.min_size_ratio + 1
        # When in testing mode we transform input features to one hot encoded
        self.test_mode = test
        self.bounds = bounds
        self.design = lsm_design

    def _get_output_cols(self):
        return kOUTPUT_FEATS

    def _get_input_cols(self) -> list[str]:
        feats: list[str] = kINPUT_FEATS_DICT[self.design]
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
        if self.design == Policy.Classic:
            return one_hot_lcm_classic(data, self.categories)
        elif self.design == Policy.QFixed:
            return one_hot_lcm(data, num_feat, 2, self.categories)
        elif self.design == Policy.KHybrid:
            return one_hot_lcm(data, num_feat, self.max_levels + 1, self.categories)
        elif self.design == Policy.YZHybrid:
            return one_hot_lcm(data, num_feat, 3, self.categories)
        elif self.design in [Policy.Leveling, Policy.Tiering]:
            raise NotImplementedError
        else:
            raise TypeError("Incompatible LSM design")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["T"] = df["T"] - self.min_size_ratio
        if self.design == Policy.QFixed:
            df["Q"] -= self.min_size_ratio - 1
        elif self.design == Policy.YZHybrid:
            df["Y"] -= self.min_size_ratio - 1
            df["Z"] -= self.min_size_ratio - 1
        elif self.design == Policy.KHybrid:
            for i in range(self.max_levels):
                df[f"K_{i}"] -= self.min_size_ratio - 1
                df[f"K_{i}"] = df[f"K_{i}"].clip(lower=0)
        elif self.design in (Policy.Leveling, Policy.Tiering, Policy.Classic):
            pass

        return df

    def __iter__(self):
        files = self._fnames
        if self._shuffle:
            np.random.shuffle(files)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            file_bins = np.array_split(self._fnames, worker_info.num_workers)
            files = file_bins[worker_info.id]

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