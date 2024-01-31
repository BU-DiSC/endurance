import glob
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pa
import torch
import torch.utils.data

from endure.ltune.data.input_features import kINPUT_FEATS


class LTuneDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self,
        folder: str,
        format: str = "parquet",
        shuffle: bool = False,
    ) -> None:
        self._format = format
        self._fnames = glob.glob(os.path.join(folder, "*." + format))
        self._shuffle = shuffle

    def _get_input_cols(self):
        return kINPUT_FEATS

    def _load_data(self, fname):
        if self._format == "parquet":
            df = pa.read_table(fname).to_pandas()
        else:
            df = pd.read_csv(fname)

        return df

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self._fnames
        else:
            file_bins = np.array_split(self._fnames, worker_info.num_workers)
            files = file_bins[worker_info.id]
            if self._shuffle:
                np.random.shuffle(files)
        for file in files:
            df = self._load_data(file)
            inputs = torch.from_numpy(df[self._get_input_cols()].values).float()
            for input in inputs:
                # We return input twice, as the loss function for our learned
                # tuner will be taking in the workload as a label, and the
                # design prediciton to calculate estimate IO cost
                yield input, input
            del df  # attempt to release dataframe memory
