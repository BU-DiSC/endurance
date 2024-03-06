import torch
import numpy as np
from typing import List, Optional, Tuple
import logging
import os
import time
from itertools import product

from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from botorch.models.transforms import Normalize, Standardize

from endure.lsm.cost import EndureCost
from endure.data.io import Reader
from endure.lsm.types import LSMDesign, System, Policy, Workload, LSMBounds, STR_POLICY_DICT
from endure.lcm.data.generator import ClassicGenerator, QCostGenerator, YZCostGenerator, KHybridGenerator
from endure.lsm.solver.classic_solver import ClassicSolver
from endure.lsm.solver.qlsm_solver import QLSMSolver
from endure.lsm.solver.yzlsm_solver import YZLSMSolver
from endure.lsm.solver.klsm_solver import KLSMSolver
from jobs.infra.db_log import initialize_database, log_new_run, log_design_cost, log_run_details


def print_best_designs(best_designs: List[Tuple[LSMDesign, float]]) -> None:
    sorted_designs = sorted(best_designs, key=lambda x: x[1])
    print("Best Design Found:")
    for design, cost in sorted_designs[:1]:
        if design.policy == Policy.KHybrid:
            k_values_str = ", ".join(str(k) for k in design.K)
            print(f"Design: h={design.h}, T={design.T}, Policy={design.policy}, K=[{k_values_str}], Cost={cost}")
        else:
            print(f"Design: h={design.h}, T={design.T}, Policy={design.policy}, Q={design.Q}, Y={design.Y},"
                  f" Z={design.Z}, Cost={cost}")
    with open('best_designs.txt', 'w') as file:
        file.write("All Best Designs Found:\n")
        for design, cost in best_designs:
            file.write(f"Design: h={design.h}, T={design.T}, Policy={design.policy}, Q={design.Q}, "
                       f"Y={design.Y}, Z={design.Z}, Cost={cost}\n")


class BayesianPipeline:
    def __init__(self, conf: dict) -> None:
        self.end_time = None
        self.start_time = None
        self.config: dict = conf
        self.bayesian_setting: dict = self.config["job"]["BayesianOptimization"]
        self.bounds = LSMBounds()
        self.max_levels = self.bounds.max_considered_levels
        self.cf: EndureCost = EndureCost(self.max_levels)
        # self.log: logging.Logger = logging.getLogger(self.config["log"]["name"])

        self.system: System = System(**self.bayesian_setting["system"])
        self.workload: Workload = Workload(**self.bayesian_setting["workload"])
        self.initial_samples: int = self.bayesian_setting["initial_samples"]
        self.acquisition_function: str = self.bayesian_setting["acquisition_function"]
        self.q: int = self.bayesian_setting["batch_size"]
        self.num_restarts: int = self.bayesian_setting["num_restarts"]
        self.raw_samples: int = self.bayesian_setting["raw_samples"]
        self.num_iterations: int = self.bayesian_setting["num_iterations"]
        self.beta_value: float = self.bayesian_setting["beta_value"]
        self.conn = None
        self.run_id: int = 0
        self.write_to_db = self.bayesian_setting["database"]["write_to_db"]
        self.output_dir = os.path.join(
            self.bayesian_setting["database"]["data_dir"], self.bayesian_setting["database"]["db_path"]
        )
        self.db_path = os.path.join(self.output_dir, self.bayesian_setting["database"]["db_name"])
        model_type_str = self.bayesian_setting.get('model_type', 'Classic')
        self.model_type = STR_POLICY_DICT.get(model_type_str, Policy.Classic)
        self.num_k_values = self.bayesian_setting["num_k_values"]

    def run(self, system: Optional[System] = None, workload: Optional[Workload] = None, num_iterations: Optional[int] = None,
            sample_size: Optional[int] = None, acqf: Optional[str] = None) -> Tuple[Optional[LSMDesign], Optional[float]]:
        self.start_time = time.time()
        self.initialize_environment(system, workload, num_iterations, sample_size, acqf)
        train_x, train_y, best_y = self._generate_initial_data(self.initial_samples)
        best_designs = self.optimization_loop(train_x, train_y, best_y)
        best_design, best_cost, elapsed_time = self.finalize_optimization(best_designs)
        return best_design, best_cost

    def initialize_environment(self, system: Optional[System], workload: Optional[Workload], num_iterations: Optional[int],
                               sample_size: Optional[int], acqf: Optional[str]):
        os.makedirs(self.output_dir, exist_ok=True)
        self.conn = initialize_database(self.db_path)
        self.system = system if system is not None else self.system
        self.initial_samples = sample_size if sample_size is not None else self.initial_samples
        self.workload = workload if workload is not None else self.workload
        self.acquisition_function = acqf if acqf is not None else self.acquisition_function
        self.num_iterations = num_iterations if num_iterations is not None else self.num_iterations
        self.run_id = log_new_run(self.conn, self.system, self.workload, self.num_iterations,
                                  self.initial_samples, self.acquisition_function)

    def generate_initial_bounds(self, system: System) -> torch.Tensor:
        h_bounds = torch.tensor([self.bounds.bits_per_elem_range[0], min(np.floor(system.H)
                                                                         , self.bounds.bits_per_elem_range[1])], dtype=torch.float)
        t_bounds = torch.tensor([self.bounds.size_ratio_range[0], self.bounds.size_ratio_range[1]])
        policy_bounds = torch.tensor([0, 1])
        if self.model_type == Policy.QFixed:
            q_bounds = torch.tensor([1, self.bounds.size_ratio_range[1] - 1])
            bounds = torch.stack([h_bounds, t_bounds, q_bounds], dim=-1)
        elif self.model_type == Policy.YZHybrid:
            y_bounds = torch.tensor([1, self.bounds.size_ratio_range[1] - 1])
            z_bounds = torch.tensor([1, self.bounds.size_ratio_range[1] - 1])
            bounds = torch.stack([h_bounds, t_bounds, y_bounds, z_bounds], dim=-1)
        elif self.model_type == Policy.KHybrid:
            lower_limits = [self.bounds.bits_per_elem_range[0], self.bounds.size_ratio_range[0]] +\
                           [1] * self.num_k_values
            upper_limits = [min(np.floor(system.H), self.bounds.bits_per_elem_range[1]),
                            self.bounds.size_ratio_range[1]] + \
                           [self.bounds.size_ratio_range[1] - 1] * self.num_k_values
            new_bounds_list = [lower_limits, upper_limits]
            bounds = torch.tensor(new_bounds_list, dtype=torch.float64)
        else:
            bounds = torch.stack([h_bounds, t_bounds, policy_bounds], dim=-1)
        return bounds

    def optimization_loop(self, train_x, train_y, best_y):
        bounds = self.generate_initial_bounds(self.system)
        fixed_feature_list = self._initialize_feature_list(bounds)
        best_designs = []
        for i in range(self.num_iterations):
            new_candidates = self.get_next_points(train_x, train_y, best_y, bounds, fixed_feature_list,
                                                  self.acquisition_function, 1)
            new_designs, costs = self.evaluate_new_candidates(new_candidates)
            train_x, train_y, best_y, best_designs = self.update_training_data(train_x, train_y, new_candidates, costs,
                                                                               best_designs)
        #     self.log.debug(f"Iteration {i + 1}/{self.num_iterations} complete")
        # self.log.debug("Bayesian Optimization completed")
        return best_designs

    def _initialize_feature_list(self, bounds):
        t_bounds = bounds[:, 1]
        lower_t_bound = int(np.floor(t_bounds[0].item()))
        upper_t_bound = int(np.ceil(t_bounds[1].item()))
        fixed_features_list = []
        if self.model_type == Policy.Classic:
            for size_ratio in range(lower_t_bound, upper_t_bound + 1):
                for pol in range(2):
                    fixed_features_list.append({1: size_ratio, 2: pol})
        elif self.model_type == Policy.QFixed:
            for size_ratio in range(lower_t_bound, upper_t_bound + 1):
                for q in range(1, size_ratio - 1):
                    fixed_features_list.append({1: size_ratio, 2: q})
        elif self.model_type == Policy.YZHybrid:
            for size_ratio in range(lower_t_bound, upper_t_bound + 1):
                for y in range(1, size_ratio - 1):
                    for z in range(1, size_ratio - 1):
                        fixed_features_list.append({1: size_ratio, 2: y, 3: z})
        elif self.model_type == Policy.KHybrid:
            for t in range(2, upper_t_bound + 1):
                param_values = [range(1, upper_t_bound)] * self.num_k_values
                for combination in product(*param_values):
                    fixed_feature = {1: t}
                    fixed_feature.update({i + 2: combination[i] for i in range(len(combination))})
                    fixed_features_list.append(fixed_feature)
        return fixed_features_list

    def evaluate_new_candidates(self, new_candidates):
        new_designs = self.create_designs_from_candidates(new_candidates)

        costs = [self.cf.calc_cost(design, self.system, self.workload.z0, self.workload.z1, self.workload.q,
                                   self.workload.w) for design in new_designs]
        for design, cost in zip(new_designs, costs):
            log_design_cost(self.conn, self.run_id, design, cost)

        return new_designs, costs

    def update_training_data(self, train_x, train_y, new_candidates, costs, best_designs):
        new_target = torch.tensor(costs).unsqueeze(-1)
        train_x = torch.cat([train_x, new_candidates])
        train_y = torch.cat([train_y, new_target])
        best_y = train_y.min().item()
        best_designs = self._update_best_designs(best_designs, new_candidates, new_target)
        return train_x, train_y, best_y, best_designs

    def create_designs_from_candidates(self, candidates):
        for candidate in candidates:
            new_designs = self._generate_new_designs_helper(candidate)
        return new_designs

    def _generate_new_designs_helper(self, candidate):
        new_designs = []
        h = candidate[0].item()
        if h == self.system.H:
            h = h - 0.01
        if self.model_type == Policy.QFixed:
            size_ratio, q_val = candidate[1].item(), candidate[2].item()
            policy = Policy.QFixed
            new_designs = [LSMDesign(h=h, T=np.ceil(size_ratio), policy=policy, Q=int(q_val))]
            # Uncomment the following lines of code if you want the q value to be the same
            # through all levels and behave like KLSM
            # policy = Policy.KHybrid
            k_values = [q_val for _ in range(1, self.max_levels)]
            new_designs = [LSMDesign(h=h, T=np.ceil(size_ratio), policy=policy, K=k_values)]
        elif self.model_type == Policy.YZHybrid:
            size_ratio, y_val, z_val = candidate[1].item(), candidate[2].item(), candidate[3].item()
            policy = Policy.YZHybrid
            new_designs = [LSMDesign(h=h, T=np.ceil(size_ratio), policy=policy, Y=int(y_val), Z=int(z_val))]
        elif self.model_type == Policy.KHybrid:
            size_ratio = candidate[1].item()
            k_values = [cand.item() for cand in candidate[2:]]
            policy = Policy.KHybrid
            if len(k_values) < self.max_levels:
                k_values += [1] * (self.max_levels - len(k_values))
            new_designs.append(LSMDesign(h=h, T=np.ceil(size_ratio), policy=policy, K=k_values))
        else:
            size_ratio, policy_val = candidate[1].item(), candidate[2].item()
            policy = Policy.Leveling if policy_val < 0.5 else Policy.Tiering
            new_designs = [LSMDesign(h, np.ceil(size_ratio), policy)]

        return new_designs

    def finalize_optimization(self, best_designs):
        elapsed_time = time.time() - self.start_time
        sorted_designs = sorted(best_designs, key=lambda x: x[1])
        analaytical_design, analytical_cost = self._find_analytical_results(self.system,
                                                                            self.workload.z0, self.workload.z1,
                                                                            self.workload.q, self.workload.w)
        best_design, best_cost = sorted_designs[0][0], sorted_designs[0][1]
        log_run_details(self.conn, self.run_id, elapsed_time, analytical_cost, best_cost, analaytical_design,
                        best_design)
        self.conn.close()
        if sorted_designs:
            return best_design, best_cost, elapsed_time
        else:
            return None, None, elapsed_time

    def get_next_points(self, x: torch.Tensor, y: torch.Tensor, best_y: float, bounds: torch.Tensor,
                        fixed_features_list: List, acquisition_function: str = "ExpectedImprovement", n_points: int = 1, ) -> torch.Tensor:
        if self.model_type == Policy.QFixed or self.model_type == Policy.Classic:
            single_model = MixedSingleTaskGP(x, y, cat_dims=[1, 2], input_transform=Normalize(d=x.shape[1],
                                                                                              bounds=bounds),
                                             outcome_transform=Standardize(m=1))
        elif self.model_type == Policy.YZHybrid:
            single_model = MixedSingleTaskGP(x, y, cat_dims=[1, 2, 3], input_transform=Normalize(d=x.shape[1],
                                                                                                 bounds=bounds),
                                             outcome_transform=Standardize(m=1))
        elif self.model_type == Policy.KHybrid:
            # the self.num_k_values represents the number of categorical values the model
            # is predicting out of the self.max_levels. The +2 is because this is the list of indices
            # and the first 2 indices represent the 'h' value and then the 'T'value. So everything from index 1
            # till the size of num_k_values + 2 is a categorical value
            cat_dims = list(range(1, self.num_k_values + 2))
            single_model = MixedSingleTaskGP(x, y, cat_dims=cat_dims, input_transform=Normalize(d=x.shape[1],
                                                                                                bounds=bounds),
                                             outcome_transform=Standardize(m=1))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
        fit_gpytorch_model(mll)
        if acquisition_function == "ExpectedImprovement":
            acqf = ExpectedImprovement(model=single_model, best_f=best_y, maximize=False)
        elif acquisition_function == "UpperConfidenceBound":
            beta = self.beta_value
            acqf = UpperConfidenceBound(model=single_model, beta=beta, maximize=False)
        elif acquisition_function == "qExpectedImprovement":
            acqf = qExpectedImprovement(model=single_model, best_f=-best_y)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
        candidates, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=n_points,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            fixed_features_list=fixed_features_list
        )
        return candidates

    def _generate_initial_data(self, n: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        train_x = []
        train_y = []
        if self.model_type == Policy.QFixed:
            generator = QCostGenerator(self.bounds)
        elif self.model_type == Policy.YZHybrid:
            generator = YZCostGenerator(self.bounds)
        elif self.model_type == Policy.KHybrid:
            generator = KHybridGenerator(self.bounds)
        else:
            generator = ClassicGenerator(self.bounds)
        for _ in range(n):
            design = generator._sample_design(self.system)
            if self.model_type == Policy.Classic:
                if design.policy == Policy.Leveling:
                    policy = 0
                elif design.policy == Policy.Tiering:
                    policy = 1
                x_values = np.array([design.h, design.T, policy])
            elif self.model_type == Policy.QFixed:
                x_values = np.array([design.h, design.T, design.Q])
            elif self.model_type == Policy.YZHybrid:
                x_values = np.array([design.h, design.T, design.Y, design.Z])
            elif self.model_type == Policy.KHybrid:
                k_values_padded = (design.K + [1] * self.num_k_values)[:self.num_k_values]
                x_values = np.array([design.h, design.T] + k_values_padded)
            cost = self.cf.calc_cost(design, self.system, self.workload.z0, self.workload.z1,
                                     self.workload.q, self.workload.w)
            log_design_cost(self.conn, self.run_id, design, cost)
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
            if self.model_type == Policy.QFixed:
                h, size_ratio, qvalue = x[0], x[1], x[2]
                best_designs.append((LSMDesign(h=h.item(), T=np.ceil(size_ratio.item()), policy=Policy.QFixed,
                                               Q=qvalue.item()), y.item()))
            elif self.model_type == Policy.YZHybrid:
                h, size_ratio, yvalue, zvalue = x[0], x[1], x[2], x[3]
                best_designs.append(
                    (LSMDesign(h=h.item(), T=np.ceil(size_ratio.item()), policy=Policy.YZHybrid, Y=yvalue.item(),
                               Z=zvalue.item()), y.item()))
            elif self.model_type == Policy.KHybrid:
                h, size_ratio = x[0], x[1]
                k_values = x[2:].tolist()
                design = LSMDesign(h=h.item(), T=np.ceil(size_ratio.item()), policy=Policy.KHybrid, K=k_values)
                best_designs.append((design, y.item()))
            else:
                h, size_ratio, policy = x[0], x[1], x[2]
                if policy.item() < 0.5:
                    pol = Policy.Leveling
                else:
                    pol = Policy.Tiering
                best_designs.append((LSMDesign(h=h.item(), T=np.ceil(size_ratio.item()), policy=pol), y.item()))
        return best_designs

    def _find_analytical_results(self, system: System, z0: float, z1: float, q: float, w: float,
                                 conf: Optional[dict] = None) -> Tuple[LSMDesign, float]:
        conf = conf if conf is not None else self.config
        if self.model_type == Policy.Classic:
            solver = ClassicSolver(conf)
        elif self.model_type == Policy.QFixed:
            solver = QLSMSolver(conf)
        elif self.model_type == Policy.YZHybrid:
            solver = YZLSMSolver(conf)
        elif self.model_type == Policy.KHybrid:
            solver = KLSMSolver(conf)
        nominal_design, nominal_solution = solver.get_nominal_design(system, z0, z1, q, w)

        if self.model_type == Policy.Classic:
            x = np.array([[nominal_design.h, nominal_design.T]])
            policy = nominal_design.policy
            cost = solver.nominal_objective(x[0], policy, system, z0, z1, q, w)
        elif self.model_type == Policy.QFixed:
            x = np.array([[nominal_design.h, nominal_design.T, nominal_design.Q]])
            cost = solver.nominal_objective(x[0], system, z0, z1, q, w)
        elif self.model_type == Policy.YZHybrid:
            x = np.array([[nominal_design.h, nominal_design.T, nominal_design.Y, nominal_design.Z]])
            cost = solver.nominal_objective(x[0], system, z0, z1, q, w)
        elif self.model_type == Policy.KHybrid:
            k_values = nominal_design.K
            x = np.array([nominal_design.h, nominal_design.T] + k_values)
            cost = solver.nominal_objective(x, system, z0, z1, q, w)
        print("Cost for the nominal design using analytical solver: ", cost)
        print("Nominal Design suggested by analytical solver: ", nominal_design)
        return nominal_design, cost


if __name__ == "__main__":
    config = Reader.read_config("endure.toml")

    log = logging.getLogger(config["log"]["name"])
    log.info("Initializing Bayesian Optimization Job")

    bayesian_optimizer = BayesianPipeline(config)
    bayesian_optimizer.run()
