import sys
import os
import csv
import toml
import torch

sys.path.append(os.path.join(sys.path[0], "../../"))

from axe.lsm.types import LSMBounds, Workload
from axe.lcm.data.generator import LCMDataGenerator
from jobs.bayesian_pipeline import BayesianPipeline
from axe.lsm.cost import EndureCost


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
            if attr_name.startswith("__"):
                continue

            try:
                attr_value = getattr(obj, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(obj, attr_name, attr_value.to(device))
                elif hasattr(attr_value, "__dict__") or isinstance(
                    attr_value, (list, dict)
                ):
                    to_cuda(attr_value, seen)
            except Exception:
                pass
    else:
        print("CUDA not available")


def compare_designs(n_runs=100, csv_filename="yz_design_comparison.csv"):
    """Compare Bayesian and analytical designs."""
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Entries per page(E)",
                "Physical Entries per page(B)",
                "Selectivity(s)",
                "Max bits per element(H)",
                "Total elements (N)",
                "Empty Reads",
                "Non-Empty Reads",
                "Range Queries",
                "Writes",
                "BO Design",
                "Analytical Design",
                "BO Cost",
                "Analytical Cost",
                "Diff(Analytical-Bayesian)",
            ]
        )

        for i in range(n_runs):
            print(f"Iteration {i + 1}/{n_runs} running")
            system = generator._sample_system()
            z0, z1, q, w = generator._sample_workload(4)
            workload = Workload(z0=z0, z1=z1, q=q, w=w)
            bo_design, bo_cost = bayesian_optimizer.run(system, workload)
            analytical_design, analytical_cost = (
                bayesian_optimizer._find_analytical_results(system, workload)
            )
            writer.writerow(
                [
                    system.E,
                    system.B,
                    system.s,
                    system.H,
                    system.N,
                    workload.z0,
                    workload.z1,
                    workload.q,
                    workload.w,
                    bo_design,
                    analytical_design,
                    bo_cost,
                    analytical_cost,
                    analytical_cost - bo_cost,
                ]
            )


if __name__ == "__main__":
    file_dir = os.path.dirname(__file__)
    config_path = os.path.join(file_dir, "axe.toml")
    with open(config_path) as fid:
        config = toml.load(fid)
    bayesian_optimizer = BayesianPipeline(config)
    bounds = LSMBounds(**config["lsm"]["bounds"])
    generator = LCMDataGenerator(bounds)
    cf = EndureCost(max_levels=bounds.max_considered_levels)

    to_cuda(bayesian_optimizer)
    to_cuda(generator)
    to_cuda(cf)
    compare_designs(
        config["job"]["BayesianOptimization"]["multi_jobs_number"],
        config["job"]["BayesianOptimization"]["multi_job_file"],
    )
