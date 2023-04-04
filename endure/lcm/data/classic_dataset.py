import os
import logging
import glob
import torch
import numpy as np
import pandas as pd
import torchdata.datapipes as DataPipe


class LCMDataSet(torch.utils.data.Dataset):
    def __init__(self, config, folder, format='csv'):
        self._config = config
        self.log = logging.getLogger(config['log']['name'])
        self._mean = np.array(config['lcm']['data']['mean_bias'], np.float32)
        self._std = np.array(config['lcm']['data']['std_bias'], np.float32)
        self._format = format

        fnames = glob.glob(os.path.join(folder, '*.' + self._format))
        self.log.info('Loading in all files to RAM')
        self._df = self._load_data(fnames)
        self._label_cols = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost']
        self._input_cols = self._get_input_cols()

        self.inputs = torch.from_numpy(
            self._df[self._input_cols].values).float()
        self.labels = torch.from_numpy(
            self._df[self._label_cols].values).float()

    def _get_input_cols(self):
        base = ['z0', 'z1', 'q', 'w', 'h', 'T']
        choices = {
            'KLSM': [f'K_{i}'
                     for i in range(self._config['lsm']['max_levels'])],
            'QLSM': ['Q'],
            'Tier': [],
            'Level': [],
        }
        extension = choices.get(self._config['lsm']['design'], None)
        if extension is None:
            self.log.warn('Invalid model defaulting to KCost')
            extension = choices.get('KCost')

        return base + extension

    def _load_data(self, fnames):
        df = []
        for fname in fnames:
            df.append(pd.read_csv(fname))
        df = pd.concat(df)

        return self._process_df(df)

    def _process_df(self, df):
        df[['z0', 'z1', 'q', 'w', 'h']] -= self._mean
        df[['z0', 'z1', 'q', 'w', 'h']] /= self._std
        df['T'] = df['T'] - self._config['lsm']['size_ratio']['min']
        if self._config['lsm']['design'] == 'QLSM':
            df['Q'] -= (self._config['lsm']['size_ratio']['min'] - 1)
        elif self._config['lsm']['design'] in ['Tier', 'Level']:
            pass
        elif self._config['lsm']['design'] == 'KLSM':
            for i in range(self._config['lsm']['max_levels']):
                df[f'K_{i}'] -= (self._config['lsm']['size_ratio']['min'] - 1)
                df[f'K_{i}'][df[f'K_{i}'] < 0] = 0
        else:
            self.log.warn('Invalid model defaulting to KCost behavior')
            for i in range(self._config['lsm']['max_levels']):
                df[f'K_{i}'] -= (self._config['lsm']['size_ratio']['min'] - 1)
                df[f'K_{i}'][df[f'K_{i}'] < 0] = 0

        return df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        return self.labels[idx], self.inputs[idx]


class LCMDataPipeGenerator():
    def __init__(self, config):
        self._config = config
        self.mean = np.array(config['lcm']['data']['mean_bias'], np.float32)
        self.std = np.array(config['lcm']['data']['std_bias'], np.float32)

    def _process_row(self, row):
        labels = np.array(row[0:4], np.float32)
        features = np.array(row[4:], np.float32)

        # First 5 are z0, z1, w, q, h
        # TODO: Streamline this process
        continuous_data = features[0:5]
        continuous_data -= self.mean
        continuous_data /= self.std

        # Remaining will be T and Ks
        categorical_data = features[5:]
        categorical_data[0] -= self._config['lsm']['size_ratio']['min']
        categorical_data[1:] -= (self._config['lsm']['size_ratio']['min'] - 1)
        features = np.concatenate((continuous_data, categorical_data))

        return (labels, features)

    def build_dp(self, folder, shuffle=True) -> DataPipe.iter.IterDataPipe:
        dp = DataPipe.iter.FileLister(folder)
        dp = dp.filter(filter_fn=lambda fname: fname.endswith('.csv'))
        dp = DataPipe.iter.FileOpener(dp, mode='rt')
        dp = dp.parse_csv(delimiter=',', skip_lines=1)
        dp = dp.map(self._process_row)
        dp = dp.in_memory_cache()
        if shuffle:
            dp = dp.shuffle()
        dp = dp.sharding_filter()

        return dp
