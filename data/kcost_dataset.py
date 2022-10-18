import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
import torchdata.datapipes as DataPipe


class KCostDataSetSplit(Dataset):
    def __init__(self, config, paths: list[str]):
        self.config = config
        df = []
        for path in paths:
            df.append(pd.read_feather(path))
        df = pd.concat(df)

        cont_inputs = ['h', 'z0', 'z1', 'q', 'w']
        cate_inputs = ['T']
        for i in range(config['static_params']['max_levels']):
            cate_inputs += [f'K_{i}']
        output_cols = ['k_cost']

        self.cont_inputs = torch.from_numpy(df[cont_inputs].values).float()
        self.cate_inputs = torch.from_numpy(df[cate_inputs].values)

        self.outputs = torch.from_numpy(df[output_cols].values).float()

    def __len__(self):
        return len(self.cont_inputs)

    def __getitem__(self, idx):
        categories = torch.flatten(
                nn.functional.one_hot(
                    self.cate_inputs[idx],
                    num_classes=self.config['static_params']['max_size_ratio']),
                start_dim=-2)
        inputs = torch.cat((self.cont_inputs[idx], categories), dim=-1)
        label = self.outputs[idx]

        return inputs, label


class KCostDataSetSplitDF(Dataset):
    def __init__(self, config, df):
        self.config = config

        cont_inputs = ['h', 'z0', 'z1', 'q', 'w']
        cate_inputs = ['T']
        for i in range(config['static_params']['max_levels']):
            cate_inputs += [f'K_{i}']
        output_cols = ['k_cost']

        self.cont_inputs = torch.from_numpy(df[cont_inputs].values).float()
        self.cate_inputs = torch.from_numpy(df[cate_inputs].values)

        self.outputs = torch.from_numpy(df[output_cols].values).float()

    def __len__(self):
        return len(self.cont_inputs)

    def __getitem__(self, idx):
        categories = torch.flatten(
                nn.functional.one_hot(
                    self.cate_inputs[idx],
                    num_classes=self.config['static_params']['max_size_ratio']),
                start_dim=-2)
        inputs = torch.cat((self.cont_inputs[idx], categories), dim=-1)
        label = self.outputs[idx]

        return inputs, label


class KCostDataSet(Dataset):
    def __init__(self, config, paths: list[str]):
        self.config = config
        df = []
        for path in paths:
            df.append(pd.read_feather(path))
        df = pd.concat(df)

        cont_inputs = ['h', 'z0', 'z1', 'q', 'w']
        cate_inputs = ['T']
        for i in range(config['static_params']['max_levels']):
            cate_inputs += [f'K_{i}']
        output_cols = ['z0_cost', 'z1_cost', 'q_cost', 'w_cost']

        self.cont_inputs = torch.from_numpy(df[cont_inputs].values).float()
        self.cate_inputs = torch.from_numpy(df[cate_inputs].values)

        self.outputs = torch.from_numpy(df[output_cols].values).float()

    def __len__(self):
        return len(self.cont_inputs)

    def __getitem__(self, idx):
        categories = torch.flatten(
                nn.functional.one_hot(
                    self.cate_inputs[idx],
                    num_classes=self.config['static_params']['max_size_ratio']),
                start_dim=-2)
        inputs = torch.cat((self.cont_inputs[idx], categories), dim=-1)
        label = self.outputs[idx]

        return inputs, label


class KCostDataPipeGenerator():
    def __init__(self, config):
        self.config = config

    @staticmethod
    def build_datapipe(data_dir, file_filter, process_fn):
        datapipe = (DataPipe
                    .iter
                    .FileLister(data_dir)
                    .filter(filter_fn=file_filter)
                    .open_files(mode='rt')
                    .parse_csv(delimiter=',', skip_lines=1)
                    .shuffle()
                    .map(process_fn))
        return datapipe
