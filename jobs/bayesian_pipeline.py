import torch
import numpy as np
from typing import List, Optional, Tuple
import logging
import csv
import os
import time

from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from botorch.models.transforms import Normalize, Standardize

from endure.lsm.cost import EndureCost
from endure.data.io import Reader
from endure.lsm.types import LSMDesign, System, Policy, Workload
from endure.lcm.data.generator import ClassicGenerator, QCostGenerator, YZCostGenerator, KHybridGenerator
from endure.lsm.solver.classic_solver import ClassicSolver
from endure.lsm.solver.qlsm_solver import QLSMSolver
from endure.lsm.solver.yzlsm_solver import YZLSMSolver
from endure.lsm.solver.klsm_solver import KLSMSolver
from endure.util.db_log import initialize_database, log_new_run, log_design_cost


class BayesianPipeline:
    def __init__(self, conf: dict) -> None:
        self.end_time = None
        self.config: dict = conf
        self.bayesian_setting: dict = self.config["job"]["BayesianOptimization"]
        max_levels = self.config['job']['BayesianOptimization']['max_levels']
        self.cf: EndureCost = EndureCost(max_levels)
        self.log: logging.Logger = logging.getLogger(self.config["log"]["name"])

        self.system: System = System(**self.bayesian_setting["system"])
        self.workload: Workload = Workload(**self.bayesian_setting["workload"])
        self.h_bounds: torch.Tensor = torch.tensor([self.bayesian_setting["bounds"]["h_min"],
                                                    self.bayesian_setting["system"]["H"]])
        self.T_bounds: torch.Tensor = torch.tensor([self.bayesian_setting["bounds"]["T_min"],
                                                    self.bayesian_setting["bounds"]["T_max"]])
        self.policy_bounds: torch.Tensor = torch.tensor([0.0, 1.0])
        self.bounds: torch.Tensor = torch.stack([self.h_bounds, self.T_bounds, self.policy_bounds], dim=-1)
        self.initial_samples: int = self.bayesian_setting["initial_samples"]
        self.acquisition_function: str = self.bayesian_setting["acquisition_function"]
        self.q: int = self.bayesian_setting["batch_size"]
        self.num_restarts: int = self.bayesian_setting["num_restarts"]
        self.raw_samples: int = self.bayesian_setting["raw_samples"]
        self.num_iterations: int = self.bayesian_setting["num_iterations"]
        self.beta_value: float = self.bayesian_setting["beta_value"]
        self.conn = None
        self.run_id: int = None
        self.write_to_db = self.bayesian_setting["database"]["write_to_db"]
        self.output_dir = os.path.join(
            self.config["io"]["data_dir"], self.bayesian_setting["database"]["db_path"]
        )
        self.db_path = os.path.join(self.output_dir, self.bayesian_setting["database"]["db_name"])
        self.model_type = self.bayesian_setting['model_type']

    def run(self, system: Optional[System] = None, z0: Optional[float] = None, z1: Optional[float] = None,
            q: Optional[float] = None, w: Optional[float] = None, num_iterations: Optional[int] = None,
            sample_size: Optional[int] = None, acqf: Optional[str] = None) -> Tuple[Optional[LSMDesign],
                                                                                    Optional[float]]:
        start_time = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        self.conn = initialize_database(self.db_path)
        system = system if system is not None else self.system
        # print(system.E, "E - system")
        sample_size = sample_size if sample_size is not None else self.initial_samples
        z0 = z0 if z0 is not None else self.workload.z0
        z1 = z1 if z1 is not None else self.workload.z1
        q = q if q is not None else self.workload.q
        w = w if w is not None else self.workload.w
        w = w if w is not None else self.workload.w
        acqf = acqf if acqf is not None else self.acquisition_function
        iterations = num_iterations if num_iterations is not None else self.num_iterations
        workload = Workload(z0, z1, q, w)
        self.run_id = log_new_run(self.conn, system, workload, iterations, sample_size, acqf)
        train_x, train_y, best_y = self._generate_initial_data(z0, z1, q, w, system, sample_size)
        bounds = self.generate_initial_bounds(system)
        best_designs = []

        for i in range(iterations):
            new_candidates = self.get_next_points(train_x, train_y, best_y, bounds, acqf, 1)
            for cand in new_candidates:
                h = cand[0].item()
                if h == system.H:
                    h = h - 0.01
                if self.model_type == "QHybrid":
                    size_ratio, q_val = cand[1].item(), cand[2].item()
                    policy = Policy.QFixed
                    new_designs = [LSMDesign(h=h, T=np.ceil(size_ratio), policy=policy, Q=int(q_val))]
                elif self.model_type == "YZHybrid":
                    size_ratio, y_val, z_val = cand[1].item(), cand[2].item(), cand[3].item()
                    policy = Policy.YZHybrid
                    new_designs = [LSMDesign(h=h, T=np.ceil(size_ratio), policy=policy, Y=int(y_val), Z=int(z_val))]
                # TODO: Add KHybrid here
                else:
                    size_ratio, policy_val = cand[1].item(), cand[2].item()
                    policy = Policy.Leveling if policy_val < 0.5 else Policy.Tiering
                    new_designs = [LSMDesign(h, np.ceil(size_ratio), policy)]

            # This will solve the division by 0 error
            for design in new_designs:
                try:
                    self.cf.calc_cost(design, system, z0, z1, q, w)
                except ZeroDivisionError:
                    print(design, " - Design")
                    print(system, " - System")
                    print("Ratios: z0, z1, q, w: ", z0, z1, q, w)
                    raise
                except Exception as e:
                    logging.exception(e)
            costs = [self.cf.calc_cost(design, system, z0, z1, q, w) for design in new_designs]

            for design, cost in zip(new_designs, costs):
                log_design_cost(self.conn, self.run_id, design, cost)
            new_target = torch.tensor(costs).unsqueeze(-1)
            train_x = torch.cat([train_x, new_candidates])
            train_y = torch.cat([train_y, new_target])
            best_y = train_y.min().item()
            best_designs = self._update_best_designs(best_designs, new_candidates, new_target)
            self.log.debug(f"Iteration {i + 1}/{iterations} complete")
        self.log.debug("Bayesian Optimization completed")
        self._print_best_designs(best_designs)
        self._find_analytical_results(system, z0, z1, q, w) # Uncomment this if running this file through endure.py
        sorted_designs = sorted(best_designs, key=lambda x: x[1])
        self.conn.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed time", elapsed_time)
        if sorted_designs:
            best_design, best_cost = sorted_designs[0]
            return best_design, best_cost, elapsed_time
        else:
            return None, None

    def generate_initial_bounds(self, system: System) -> torch.Tensor:
        h_bounds = torch.tensor([self.bayesian_setting["bounds"]["h_min"], np.floor(system.H)])
        t_bounds = torch.tensor([int(self.bayesian_setting["bounds"]["T_min"]),
                                 int(self.bayesian_setting["bounds"]["T_max"])])
        policy_bounds = torch.tensor([0, 1])
        if self.model_type == "QHybrid":
            q_bounds = torch.tensor([1, self.bayesian_setting["bounds"]["T_max"]-1])
            bounds = torch.stack([h_bounds, t_bounds, q_bounds], dim=-1)
        elif self.model_type == "YZHybrid":
            y_bounds = torch.tensor([1, self.bayesian_setting["bounds"]["T_max"]-1])
            z_bounds = torch.tensor([1, self.bayesian_setting["bounds"]["T_max"]-1])
            bounds = torch.stack([h_bounds, t_bounds, y_bounds, z_bounds], dim=-1)
        # elif self.model_type == "KHybrid": # TODO add support for KHybrid model
        else:
            bounds = torch.stack([h_bounds, t_bounds, policy_bounds], dim=-1)
        return bounds

    def get_next_points(self, x: torch.Tensor, y: torch.Tensor, best_y: float, bounds: torch.Tensor,
                        acquisition_function: str = "ExpectedImprovement", n_points: int = 1) -> torch.Tensor:
        if self.model_type == "QHybrid" or self.model_type == "Classic":
            single_model = MixedSingleTaskGP(x, y, cat_dims=[1, 2], input_transform=Normalize(d=x.shape[1],
                                                                                              bounds=bounds),
                                             outcome_transform=Standardize(m=1))
        elif self.model_type == "YZHybrid":
            single_model = MixedSingleTaskGP(x, y, cat_dims=[1, 2, 3], input_transform=Normalize(d=x.shape[1],
                                                                                                 bounds=bounds),
                                             outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
        fit_gpytorch_model(mll)
        if acquisition_function == "ExpectedImprovement":
            acqf = ExpectedImprovement(model=single_model, best_f=best_y, maximize=False)
        elif acquisition_function == "UpperConfidenceBound":
            beta = self.beta_value
            acqf = UpperConfidenceBound(model=single_model, beta=beta, maximize=False)
        elif acquisition_function == "qExpectedImprovement":
            acqf = qExpectedImprovement(model=single_model, best_f=best_y)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
        t_bounds = bounds[:, 1]
        lower_t_bound = int(np.floor(t_bounds[0].item()))
        upper_t_bound = int(np.ceil(t_bounds[1].item()))
        fixed_features_list = []
        if self.model_type == "Classic":
            for size_ratio in range(lower_t_bound, upper_t_bound):
                for pol in range(2):
                    fixed_features_list.append({1: size_ratio, 2: pol})
        elif self.model_type == "QHybrid":
            for size_ratio in range(lower_t_bound, upper_t_bound):
                for q in range(1, size_ratio-1):
                    fixed_features_list.append({1: size_ratio, 2: q})
        elif self.model_type == "YZHybrid":
            for size_ratio in range(lower_t_bound, upper_t_bound):
                for y in range(1, size_ratio-1):
                    for z in range(1, size_ratio-1):
                        fixed_features_list.append({1: size_ratio, 2: y, 3: z})
        candidates, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=n_points,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            fixed_features_list=fixed_features_list
        )
        return candidates

    def _generate_initial_data(self, z0, z1, q, w, system: System, n: int = 30, run_id=None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        train_x = []
        train_y = []
        run_id = run_id if run_id is not None else self.run_id
        if self.model_type == "QHybrid":
            generator = QCostGenerator()
        elif self.model_type == "YZHybrid":
            generator = YZCostGenerator()
        elif self.model_type == "KHybrid":
            generator = KHybridGenerator()
        else:
            generator = ClassicGenerator()
        for _ in range(n):
            design = generator._sample_design(system)
            if self.model_type == "Classic":
                if design.policy == Policy.Leveling:
                    policy = 0
                elif design.policy == Policy.Tiering:
                    policy = 1
                x_values = np.array([design.h, design.T, policy])
                # log_design = LSMDesign(design.h, policy="", design.T, policy)
            elif self.model_type == "QHybrid":
                x_values = np.array([design.h, design.T, design.Q])
                # log_design = LSMDesign(design.h, design.T, policy=Policy.QFixed, Q=design.Q)
            elif self.model_type == "YZHybrid":
                x_values = np.array([design.h, design.T, design.Y, design.Z])
                # log_design = LSMDesign(design.h, design.T, design.Y, design.Z)
            # TODO: add logic for KHybrid
            cost = self.cf.calc_cost(design, system, z0, z1, q, w)
            log_design_cost(self.conn, run_id, design, cost)
            train_x.append(x_values)
            train_y.append(cost)
        train_x = np.array(train_x)
        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y, dtype=torch.float64).unsqueeze(-1)
        best_y = train_y.min().item()
        return train_x, train_y, best_y

    def _update_best_designs(self, best_designs: List[Tuple[LSMDesign, float]], new_x: torch.Tensor,
                             new_y: torch.Tensor) -> List[Tuple[LSMDesign, float]]:
        for x, y in zip(new_x, new_y):
            if self.model_type == "QHybrid":
                h, size_ratio, qvalue = x[0], x[1], x[2]
                best_designs.append((LSMDesign(h.item(), np.ceil(size_ratio.item()), qvalue.item()), y.item()))
            elif self.model_type == "YZHybrid":
                h, size_ratio, yvalue, zvalue = x[0], x[1], x[2], x[3]
                best_designs.append(
                    (LSMDesign(h.item(), np.ceil(size_ratio.item()), yvalue.item(), zvalue.item()), y.item()))
            # TODO: code for KHybrid to be added
            else:
                h, size_ratio, policy = x[0], x[1], x[2]
                # policy = Policy.Leveling if policy_continuous < 0.5 else Policy.Tiering
                best_designs.append((LSMDesign(h.item(), np.ceil(size_ratio.item()), policy.item()), y.item()))
        return best_designs

    def _print_best_designs(self, best_designs: List[Tuple[LSMDesign, float]]) -> None:
        sorted_designs = sorted(best_designs, key=lambda x: x[1])
        print("Best Design Found:")
        if self.model_type == "Classic":
            for design, cost in sorted_designs[:1]:
                print(f"Design: h={design.h}, T={design.T}, Policy={design.policy}, Cost={cost}")
            with open('best_designs.txt', 'w') as file:
                file.write("All Best Designs Found:\n")
                for design, cost in best_designs:
                    file.write(f"Design: h={design.h}, T={design.T}, Policy={design.policy}, Cost={cost}\n")
        if self.model_type == "QHybrid":
            for design, cost in sorted_designs[:1]:
                print(f"Design: h={design.h}, T={design.T}, Q={design.Q}, Cost={cost}")
            with open('best_designs.txt', 'w') as file:
                file.write("All Best Designs Found:\n")
                for design, cost in best_designs:
                    file.write(f"Design: h={design.h}, T={design.T}, Q={design.Q}, Cost={cost}\n")


    def _write_to_csv(self, best_designs: List[Tuple[LSMDesign, float]], system: Optional[System] = None,
                      filename: str = 'best_designs.csv') -> None:
        sorted_designs = sorted(best_designs, key=lambda x: x[1])[:1]
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entries per page(E)', 'Range query selectivity(s)', 'Entries per page(B)',
                             'Total elements(N)', 'max bits per element(H) ', 'bits per element(h)',
                             'size ratio(T)', 'Policy', 'Cost'])

            for design, cost in sorted_designs:
                system = system if system is not None else self.system
                writer.writerow(
                    [system.E, round(system.s, 2), system.B, system.N, system.H, round(design.h, 2), design.T,
                     design.policy.name, round(cost, 2)])

    def _find_analytical_results(self, system: System, z0: float, z1: float, q: float, w: float,
                                 conf: Optional[dict] = None) -> Tuple[LSMDesign, float]:
        conf = conf if conf is not None else self.config
        if self.model_type == "Classic":
            solver = ClassicSolver(conf)
        elif self.model_type == "QHybrid":
            solver = QLSMSolver(conf)
        elif self.model_type == "YZHybrid":
            solver = YZLSMSolver(conf)
        nominal_design, nominal_solution = solver.get_nominal_design(system, z0, z1, q, w)
        x = np.array([[nominal_design.h, nominal_design.T]])
        train_x = torch.tensor(x)
        policy = nominal_design.policy
        cost = solver.nominal_objective(x[0], policy, system, z0, z1, q, w)
        train_y = torch.tensor(cost, dtype=torch.float64).unsqueeze(-1)
        print("Cost for the nominal design using analytical solver: ", cost)
        print("Nominal Design suggested by analytical solver: ", nominal_design)
        return nominal_design, cost


if __name__ == "__main__":
    config = Reader.read_config("endure.toml")

    log = logging.getLogger(config["log"]["name"])
    log.info("Initializing Bayesian Optimization Job")

    bayesian_optimizer = BayesianPipeline(config)
    bayesian_optimizer.run()
