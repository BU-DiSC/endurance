import pandas as pd
import os
from sklearn.cluster import DBSCAN


def count_entries_in_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)
    return len(df)

def read_parquet(path):
    parquet_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('parquet')]
    df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    df['total_cost'] = df['z0_cost'] + df['z1_cost'] + df['q_cost'] + df['w_cost']
    return df

def apply_dbscan(df, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster'] = dbscan.fit_predict(df)
    return df

if __name__ == "__main__":
    parquet_file_path = '/data_comp/train-data/lcm/std/kcost/'
    name = 'cost_0000.parquet'
    filename = parquet_file_path + name
    num_entries = count_entries_in_parquet(filename)
    df = read_parquet(parquet_file_path)
    df_clustered = apply_dbscan(df)
    print(df_clustered['cluster'].value_counts())

