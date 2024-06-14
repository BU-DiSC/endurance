import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import pandas as pd
import os
import toml
import ConfigSpace as CS
from sklearn.cluster import KMeans
import mlos_core.optimizers
from endure.lsm.cost import EndureCost
from endure.lsm.types import LSMDesign, System, Policy, Workload, LSMBounds
from endure.lcm.data.generator import KHybridGenerator


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pk.dump(model, f)


def load_config(cfg_path):
    with open(cfg_path) as fid:
        return toml.load(fid)


def generate_workloads(num_workloads, bounds):
    workloads = []
    for _ in range(num_workloads):
        generator = KHybridGenerator(bounds)
        z0, z1, q, w = generator._sample_workload(4)
        workload = Workload(z0=z0, z1=z1, q=q, w=w)
        workloads.append(workload)
    return workloads


class LCMClusterMLOS:
    def __init__(self, workloads_to_predict, directory_path, config, num_k_values=20, sample_size=1000):
        self.workloads = workloads_to_predict
        self.directory_path = directory_path
        self.num_clusters = 10
        self.num_k_values = num_k_values
        self.sample_size = sample_size
        self.df = self.load_data()
        self.config = config
        self.bounds = LSMBounds(**self.config["lsm"]["bounds"])
        self.cf = EndureCost(self.bounds.max_considered_levels)

    def load_data(self):
        parquet_files = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path) if
                         f.endswith('cost_0000.parquet')]
        df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
        df = df.head(self.sample_size)
        df['total_cost'] = df['z0_cost'] + df['z1_cost'] + df['q_cost'] + df['w_cost']
        return df

    def define_config_space(self):
        input_space = CS.ConfigurationSpace(seed=1234)
        input_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(name='h', lower=1, upper=self.bounds.memory_budget_range[1]))
        input_space.add_hyperparameter(CS.UniformIntegerHyperparameter(name='T', lower=self.bounds.size_ratio_range[0],
                                                                       upper=self.bounds.size_ratio_range[1]))
        for i in range(self.num_k_values):
            input_space.add_hyperparameter(
                CS.UniformIntegerHyperparameter(name=f'K_{i}', lower=1, upper=self.bounds.size_ratio_range[1] - 1))
        return input_space

    def cluster_data(self):
        workload_columns = ['z0', 'z1', 'q', 'w']
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.df[workload_columns])
        self.df['cluster'] = kmeans.fit_predict(self.df[workload_columns])
        # self.df['cluster'] = kmeans.labels_
        return kmeans

    def train_bo_model_for_cluster(self, cluster_data):
        config_space = self.define_config_space()
        optimizer = mlos_core.optimizers.SmacOptimizer(parameter_space=config_space)
        design_columns = ['h', 'T'] + [f'K_{i}' for i in range(self.num_k_values)]
        target_column = 'total_cost'
        int_cols = ['T'] + [f'K_{i}' for i in range(self.num_k_values)]
        cluster_data[int_cols] = cluster_data[int_cols].astype(int)
        cfg = optimizer.suggest()
        print("cfg", cfg)
        optimizer.register(cluster_data[design_columns], cluster_data[target_column])
        return optimizer

    def create_optimizers(self):
        optimizers = {}
        for i in range(self.num_clusters):
            cluster_data = self.df[self.df['cluster'] == i]
            print("cluster data")
            print(cluster_data)
            optimizer = self.train_bo_model_for_cluster(cluster_data)
            optimizers[i] = optimizer
            print(f"model {i} done")
        print("All centroids have corresponding optimizers")
        return optimizers

    def run(self):
        designs = {}
        kmeans = self.cluster_data()
        centroids = kmeans.cluster_centers_
        optimizers = self.create_optimizers()
        for i, workload in enumerate(self.workloads):
            workload_array = np.array([workload.z0, workload.z1, workload.q, workload.w])
            distances = np.linalg.norm(centroids - workload_array, axis=1)
            idx = np.argmin(distances)
            optimizer = optimizers[idx]
            optimizer.get_best_observation()
            suggested_config = optimizer.suggest()
            h = suggested_config['h'].iloc[0] if isinstance(suggested_config['h'], pd.Series) else suggested_config['h']
            T = suggested_config['T'].iloc[0] if isinstance(suggested_config['T'], pd.Series) else suggested_config['T']
            k_values = [suggested_config[f'K_{i}'].iloc[0] if isinstance(suggested_config[f'K_{i}'], pd.Series) else
                        suggested_config[f'K_{i}'] for i in range(self.num_k_values)]
            design = LSMDesign(h=h, T=T, K=k_values)
            designs[i] = design
        return designs

    def run_multiple_workloads(self, workloads):
        for workload in workloads:
            self.run(workload)


if __name__ == "__main__":
    directory_path = '/data_comp/train-data/lcm/std/kcost'
    config_path = 'endure.toml'

    config = load_config(config_path)
    bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
    workloads = generate_workloads(5, bounds)
    lcm_cluster_mlos = LCMClusterMLOS(workloads, directory_path, config)
    pred_designs = lcm_cluster_mlos.run()
    print(pred_designs)
