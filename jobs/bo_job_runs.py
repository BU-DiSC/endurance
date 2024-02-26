import sys
import os
import csv
import toml
import numpy as np
import torch

sys.path.append(os.path.join(sys.path[0], '../'))

from endure.lcm.data.generator import LCMDataGenerator
from endure.data.io import Reader
from jobs.bayesian_pipeline import BayesianPipeline
from endure.lsm.solver.classic_solver import ClassicSolver
from endure.lsm.cost import EndureCost


def to_cuda(obj, seen=None):
    """Recursively move tensors to CUDA if available, avoiding infinite recursion."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        for attr_name in dir(obj):
            if attr_name.startswith('__'):
                continue

            try:
                attr_value = getattr(obj, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(obj, attr_name, attr_value.to(device))
                elif hasattr(attr_value, '__dict__') or isinstance(attr_value, (list, dict)):
                    to_cuda(attr_value, seen)
            except Exception as e:
                pass
    else:
        print("CUDA not available")


def compare_designs(n_runs=3, csv_filename='design_comparison.csv'):
    """Compare Bayesian and analytical designs."""
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Entries per page(E)', 'Physical Entries per page(B)', 'Selectivity(s)',
                         'Max bits per element(H)', 'Total elements (N)', 'Empty Reads', 'Non-Empty Reads',
                         'Range Queries', 'Writes', 'BO Design', 'Analytical Design', 'BO Cost',
                         'Analytical Cost', 'Diff(Analytical-Bayesian)'])

        for i in range(n_runs):
            print(f"Iteration {i + 1}/{n_runs} running")
            system = generator._sample_system()
            z0, z1, q, w = generator._sample_workload(4)
            bo_design, bo_cost = bayesian_optimizer.run(system, z0, z1, q, w)
            analytical_design, analytical_cost = bayesian_optimizer._find_analytical_results(system, z0, z1, q, w)
            writer.writerow([system.E, system.B, system.s, system.H, system.N, z0, z1, q, w,
                             bo_design, analytical_design, bo_cost, analytical_cost, analytical_cost - bo_cost])


if __name__ == "__main__":
    file_dir = os.path.dirname(__file__)
    config_path = os.path.join(file_dir, "../endure.toml")
    with open(config_path) as fid:
        config = toml.load(fid)
    bayesian_optimizer = BayesianPipeline(config)
    generator = LCMDataGenerator()
    solver = ClassicSolver(config)
    cf = EndureCost(config)

    to_cuda(bayesian_optimizer)
    to_cuda(generator)
    to_cuda(solver)
    to_cuda(cf)

    compare_designs()