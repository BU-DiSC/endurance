import os
import logging
import glob
import torch
import numpy as np
import pandas as pd
import torchdata.datapipes as DataPipe


class EndureDataSet(torch.utils.data.Dataset):
    def __init__(self, config, folder, device='cpu'):
        self._config = config
        self.log = logging.getLogger(config['log']['name'])
        self._mean = np.array(self._config['train']['mean_bias'], np.float32)
        self._std = np.array(self._config['train']['std_bias'], np.float32)

        fnames = glob.glob(os.path.join(folder, '*.csv'))
        self.log.info('Loading in all files to RAM')
        self._df = self._load_data(fnames)
        self._label_cols = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost']
        self._input_cols = self._get_input_cols()

        self.inputs = torch.from_numpy(
            self._df[self._input_cols].values).float().to(device)
        self.labels = torch.from_numpy(
            self._df[self._label_cols].values).float().to(device)

    def _get_input_cols(self):
        base = ['h', 'z0', 'z1', 'q', 'w', 'T']
        choices = {
            'KCost': [f'K_{i}'
                      for i in range(self._config['lsm']['max_levels'])],
            'TierLevelCost': [],
            'QCost': ['Q'],
        }
        extension = choices.get(self._config['model']['arch'], None)
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
        df[['h', 'z0', 'z1', 'q', 'w']] -= self._mean
        df[['h', 'z0', 'z1', 'q', 'w']] /= self._std
        df['T'] = df['T'] - self._config['lsm']['size_ratio']['min']
        if self._config['model']['arch'] == 'QCost':
            df['Q'] -= (self._config['lsm']['size_ratio']['min'] - 1)
        elif self._config['model']['arch'] == 'TierLevelCost':
            pass
        elif self._config['model']['arch'] == 'KCost':
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


class EndureDataPipeGenerator():
    def __init__(self, config):
        self.config = config
        self.mean = np.array(self.config['train']['mean_bias'], np.float32)
        self.std = np.array(self.config['train']['std_bias'], np.float32)

    def _process_row(self, row):
        labels = np.array(row[0:4], np.float32)
        features = np.array(row[4:], np.float32)

        # First 4 are h, z0, z1, w, q
        # TODO: Streamline this process
        continuous_data = features[0:5]
        continuous_data -= self.mean
        continuous_data /= self.std

        # Remaining will be T and Ks
        categorical_data = features[5:]
        categorical_data[0] -= self.config['lsm']['size_ratio']['min']
        categorical_data[1:] -= (self.config['lsm']['size_ratio']['min'] - 1)
        features = np.concatenate((continuous_data, categorical_data))

        return (labels, features)

    def build_dp(self, folder, shuffle=True):
        dp = (DataPipe
              .iter
              .FileLister(folder)
              .filter(filter_fn=lambda fname: fname.endswith('.csv'))
              .open_files(mode='rt')
              .parse_csv(delimiter=',', skip_lines=1)
              .map(self._process_row)
              .shuffle()
              .set_shuffle(shuffle)
              .sharding_filter())
        return dp
