import os
import logging
import toml
import torch
import pandas as pd
import dill


class Writer(object):

    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger('endure')

    def export_csv_file(self, df, filename):
        """
        Exports a dataframe in form of a csv file

        :param df:
        :param filename:
        """
        DATA_DIR = self.config['io']['data_dir']
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, sep=',', header=True, index=False)

    def export_dill_file(self, data, filename):
        """
        Exports data in form of a dill file

        :param data:
        :param filename:
        """
        DATA_DIR = self.config['io']['data_dir']
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'wb') as f:
            dill.dump(data, f)

    def export_figure(self, fig, figname, **kwargs):
        """
        Exports a figure file

        :param fig: Figure handle
        :param figname: Name of the figure with extension
        """
        DATA_DIR = self.config['app']['data_dir']
        filepath = os.path.join(DATA_DIR, figname, **kwargs)
        fig.savefig(filepath)


class Reader(object):
    """
    This class implements the data provider for reading various data.
    """

    def __init__(self, config):
        """Constructor

        :param config:
        """
        self.config = config
        self.data_dir = self.config['io']['data_dir']
        self.log = logging.getLogger('endure')

    @classmethod
    def read_config(cls, config_path):
        """Reads config file

        :param config_yaml_path
        """
        config = toml.load(config_path)

        return config

    def read_csv(self, filename, **kwargs):
        """Reads csv files

        :param filename:
        """
        csv_path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(csv_path,
                         header=0,
                         index_col=False,
                         **kwargs)

        return df

    def read_dill(self, filename):
        """Read dill files

        :param filename:
        """
        dill_path = os.path.join(self.data_dir, filename)

        with open(dill_path, 'rb') as f:
            data = dill.load(f)

        return data
