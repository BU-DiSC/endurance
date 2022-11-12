import os
from math import ceil

import pyarrow
import pyarrow.dataset as ds
pyarrow.set_cpu_count(pyarrow.cpu_count() * 2)
pyarrow.set_io_thread_count(pyarrow.cpu_count() * 4)


class BatchParquetDataset:
    """
    A class to lazy load batches from a parquet dataset

    Attributes:
        gs: GCSFileSystem
            Connection to a remote google storage from `gcsfs` package
        paths: list[str]
            List of GCS paths corresponding to each parquet file in the dataset
        use_threads: bool
            If True, parallelize IO operations while reading parquet files
        data: pyarrow.dataset.Dataset
            PyArrow Dataset object created from the parquet files.
        scanner: pyarrow.dataset.Scanner
            Scan operation for lazy loading the pyarrow dataset
        batches: iterator of pyarrow.RecordBatch
            Consumes scanner in record batches
        columns: list[str]
            List of columns in the dataset
        dim: int
            Number of dimensions in the dataset
        nrows: int, optional
            Number of rows in the entire dataset
        ignore_cols: list[str]
            List of column names that are neither imputed nor used for imputation purposes
        input_cols: list[str]
            List of column names that are imputed or used for imputing other columns, i.e., provided as input to GAIN
            model
        input_dim: int
            Number of columns that are either imputed or used for imputation purposes, i.e., provided as input to GAIN
            model

    Methods:
        get_columns()
            Returns a list of column in the dataset
        get_batches_per_epoch()
            Returns total number of batches in the entire dataset
        reset()
            Refreshes batches at the end of each epoch
    """

    def __init__(self, gs, paths, batch_size, use_threads=False, gain_config=None, columns=None, count_rows=False):
        """
        Constructor

        Args:
            gs: GCSFileSystem
                Connection to a remote google storage from `gcsfs` package
            paths: list[str]
                List of GCS paths corresponding to each parquet file in the dataset
            batch_size: int
                Maximum number of rows for scanned record batches
            use_threads: bool, optional
                If True, parallelize IO operations while reading parquet files (default: False)
            gain_config: dict of str:Any, optional
                Contains GAIN configuration, i.e., ignore_cols, imputation_cols, categorical_levels, etc. (default: None)
            columns: list[str], optional
                List of column names to read from the parquet files (default: None). If None, all the columns present in
                the parquet files are read
            count_rows: bool, optional
                If true, the scanner attempts to compute the total number of rows in the dataset. (default: False)
                Note: this is a very slow operation, especially if the dataset is extremely large in size and should be
                generally avoided unless there is an absolute need to know the number of rows.
        """
        self.gs = gs
        self.paths = paths
        self.use_threads = use_threads
        self.batch_size = batch_size

        self.data = ds.dataset(paths, filesystem=gs)
        self.scanner = self.data.scanner(
            columns=columns, batch_size=batch_size, use_threads=use_threads)
        self.batches = self.scanner.to_batches()

        # Dataset properties
        self.columns = columns if columns else self.get_columns()
        self.dim = len(self.columns)
        self.nrows = self.scanner.count_rows() if count_rows else None

        if gain_config:
            self.ignore_cols = gain_config['ignore_cols']
            self.input_cols = list(
                filter(lambda x: x not in self.ignore_cols, self.columns))
            self.input_dim = len(self.input_cols)
        else:
            self.ignore_cols = []
            self.input_cols = self.columns
            self.input_dim = self.dim

    def get_columns(self):
        """
        Returns a list of columns in the load data schema

        Returns:
            columns: list[str]
                List of column names in the dataset schema
        """
        schema = self.data.schema
        columns = schema.names
        return columns

    def get_batches_per_epoch(self):
        """
        Returns total number of batches in the data
        """
        num_batches = ceil(self.nrows / self.batch_size)
        return int(num_batches)

    def reset(self):
        """
        Refreshes batches at the end of each epoch
        """
        self.batches = self.scanner.to_batches()

    def __iter__(self):
        """
        Iterator
        """
        return self

    def __next__(self):
        """
        Returns a batch of tabular data

        Returns:
            (input_df, ignore_df): (pandas.DataFrame, pandas.DataFrame)
                `input_df` contains batch of observations with features that are either imputed or used for imputing other
                features, i.e., provided as input to the GAIN model
                `ignore_df` contains batch of observations with features that are neither imputed nor used for imputing
                other features. These features are not fed into the GAIN model

        """
        batch_df = next(self.batches).to_pandas()
        ignore_df = batch_df[self.ignore_cols]
        input_df = batch_df.drop(self.ignore_cols, axis=1)
        return input_df, ignore_df

    def __del__(self):
        """
        Exit all threads
        """
        os._exit(0)
