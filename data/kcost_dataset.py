import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset


class KCostDataSet(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        df = pd.read_feather(data_path)

        cont_inputs = ['h', 'z0', 'z1', 'q', 'w']
        cate_inputs = ['T'] + [f'K_{i}' for i in range(16)]
        output_cols = ['new_cost']

        mean = df[cont_inputs].mean()
        std = df[cont_inputs].std()
        std[std == 0] = 1
        df[cont_inputs] = (df[cont_inputs] - mean) / std

        cont_inputs = torch.from_numpy(df[cont_inputs].values).float()
        categories = torch.from_numpy(df[cate_inputs].values).to(torch.int64)
        categories = torch.flatten(
                nn.functional.one_hot(
                    categories,
                    num_classes=50),
                start_dim=-2)

        self.inputs = torch.cat([cont_inputs, categories], dim=1)
        self.outputs = torch.from_numpy(df[output_cols].values).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        label = self.outputs[idx]

        return inputs, label


class KCostDataSetSplit(Dataset):
    def __init__(self, config, paths: list[str]):
        self.config = config
        self.normalize_vars = (0, 0)
        df = []
        for path in paths:
            df.append(pd.read_feather(path))
        df = pd.concat(df)

        cont_inputs = ['h', 'z0', 'z1', 'q', 'w']
        cate_inputs = ['T']
        for i in range(config['static_params']['max_levels']):
            cate_inputs += [f'K_{i}']
        output_cols = ['new_cost']

        self.cont_inputs = torch.from_numpy(df[cont_inputs].values).float()
        self.cate_inputs = torch.from_numpy(df[cate_inputs].values).to(torch.int64)

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
