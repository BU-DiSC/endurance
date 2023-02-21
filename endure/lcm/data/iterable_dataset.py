import os
import logging
import glob
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pa


class LCMIterableDataSet(torch.utils.data.IterableDataset):
    def __init__(self, config, folder, format='csv', shuffle=False):
        self._config = config
        self.log = logging.getLogger(config['log']['name'])
        self._mean = np.array(config['lcm']['data']['mean_bias'], np.float32)
        self._std = np.array(config['lcm']['data']['std_bias'], np.float32)
        self._label_cols = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost']
        self._input_cols = self._get_input_cols()
        self._format = format
        self._fnames = glob.glob(os.path.join(folder, '*.' + format))
        self._shuffle = shuffle

    def _get_input_cols(self):
        base = ['h', 'z0', 'z1', 'q', 'w', 'T']
        choices = {
            'KLSM': [f'K_{i}'
                     for i in range(self._config['lsm']['max_levels'])],
            'Tier': [],
            'Level': [],
            'QLSM': ['Q'],
        }
        extension = choices.get(self._config['lsm']['design'], None)
        if extension is None:
            self.log.warn('Invalid design defaulting to Level')
            extension = choices.get('Level')

        return base + extension

    def _load_data(self, fname):
        if self._format == 'parquet':
            df = pa.read_table(fname).to_pandas()
        else:  # default csv
            df = pd.read_csv(fname)

        return self._process_df(df)

    def _process_df(self, df):
        df[['h', 'z0', 'z1', 'q', 'w']] -= self._mean
        df[['h', 'z0', 'z1', 'q', 'w']] /= self._std
        df['T'] = df['T'] - self._config['lsm']['size_ratio']['min']

        if self._config['lsm']['design'] == 'QLSM':
            df['Q'] -= (self._config['lsm']['size_ratio']['min'] - 1)
        elif self._config['lsm']['design'] == 'KLSM':
            for i in range(self._config['lsm']['max_levels']):
                df[f'K_{i}'] -= (self._config['lsm']['size_ratio']['min'] - 1)
                df[f'K_{i}'][df[f'K_{i}'] < 0] = 0
        elif self._config['lsm']['design'] in ['Tier', 'Level']:
            pass
        else:
            self.log.warn('Invalid model defaulting to KCost behavior')
            for i in range(self._config['lsm']['max_levels']):
                df[f'K_{i}'] -= (self._config['lsm']['size_ratio']['min'] - 1)
                df[f'K_{i}'][df[f'K_{i}'] < 0] = 0

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
            labels = torch.from_numpy(df[self._label_cols].values).float()
            inputs = torch.from_numpy(df[self._input_cols].values).float()
            for label, input in zip(labels, inputs):
                yield label, input
            del df  # attempt to release dataframe memory
