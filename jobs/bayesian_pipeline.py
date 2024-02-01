import torch
import numpy as np
from typing import Any, List, Union, Optional, Tuple
import logging

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

from endure.lsm.cost import EndureCost
from endure.data.io import Reader
from endure.lcm.data.generator import LCMDataGenerator
from endure.lsm.types import LSMDesign, System, Policy, Workload
from endure.lsm.solver.classic_solver import ClassicSolver


class BayesianPipeline:
    def __init__(self, config):
        self.config = config
        self.bayesian_setting = config["job"]["BayesianOptimization"]
        self.cf = EndureCost(self.config["lsm"]["max_levels"])
        self.log = logging.getLogger(config["log"]["name"])
        self.lcm_data_generator = LCMDataGenerator(config)

        self.system = System(**self.bayesian_setting["system"])
        self.workload = Workload(**self.bayesian_setting["workload"])
        self.h_bounds = torch.tensor(
            [
                self.bayesian_setting["bounds"]["h_min"],
                self.bayesian_setting["bounds"]["h_max"],
            ]
        )
        self.T_bounds = torch.tensor(
            [
                self.bayesian_setting["bounds"]["T_min"],
                self.bayesian_setting["bounds"]["T_max"],
            ]
        )
        self.bounds = torch.stack([self.h_bounds, self.T_bounds])
        self.initial_samples = self.bayesian_setting["initial_samples"]
        self.acquisition_function = self.bayesian_setting["acquisition_function"]
        self.q = self.bayesian_setting["batch_size"]
        self.num_restarts = self.bayesian_setting["num_restarts"]
        self.raw_samples = self.bayesian_setting["raw_samples"]
        self.num_iterations = self.bayesian_setting["num_iterations"]
        self.best_designs = []

        # model initial generation where:
        # train_x is the parameters we are optimizing
        # train_y is the target that we are optimizing that is minimized or maximized
        self.train_x, self.train_y = self._generate_initial_data(
            self.system, self.initial_samples
        )
        self.scaled_train_x, self.standardized_train_y = self._scale_and_standardize(
            self.train_x, self.train_y
        )
        self.mll, self.gp_model = self._initialize_model(
            self.scaled_train_x, self.standardized_train_y
        )

    def run(self) -> None:
        self.log.debug("Starting Bayesian Optimization")
        for i in range(self.num_iterations):
            new_x, new_y = self._optimize_acquisition_function(
                self.acquisition_function, self.gp_model, self.train_y
            )
            self.train_x = torch.cat([self.train_x, new_x])
            self.train_y = torch.cat([self.train_y.squeeze(), new_y])
            self.gp_model.set_train_data(
                inputs=self.train_x, targets=self.train_y, strict=False
            )
            fit_gpytorch_model(self.mll)
            self._update_best_designs(new_x, new_y)
            self.log.debug(f"Iteration {i+1}/{self.num_iterations} complete")
        self._print_best_designs()
        self._find_analytical_results()
        self.log.debug("Bayesian Optimization completed")

    def _generate_initial_data(
        self, system: System, n: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_x = []
        train_y = []
        for _ in range(n):
            design = self.lcm_data_generator._sample_design(system)
            x_values = np.array([design.h, design.T])
            # cost is negated here
            cost = -self.cf.calc_cost(
                design,
                system,
                self.workload.z0,
                self.workload.z1,
                self.workload.q,
                self.workload.w,
            )
            train_x.append(x_values)
            train_y.append(cost)
        train_x = np.array(train_x)
        train_x = torch.tensor(train_x, dtype=torch.float64)
        train_y = torch.tensor(train_y, dtype=torch.float64).unsqueeze(-1)
        return train_x, train_y

    def _scale_and_standardize(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled_train_x = self._min_max_scale(train_x)
        standardized_train_y = self._standardize_mean_std(train_y)

        return scaled_train_x, standardized_train_y

    def _min_max_scale(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.min(0, keepdim=True)[0]
        x_max = x.max(0, keepdim=True)[0]
        scaled_x = (x - x_min) / (x_max - x_min)
        return scaled_x

    def _standardize_mean_std(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean()
        x_std = x.std()
        standardized_x = (x - x_mean) / x_std
        return standardized_x

    def _initialize_model(
        self, train_x: torch.Tensor, train_y: torch.Tensor, state_dict: dict = None
    ) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
        gp_model = SingleTaskGP(train_x, train_y)
        if state_dict is not None:
            gp_model.load_state_dict(state_dict)

        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        return mll, gp_model

    def _optimize_acquisition_function(
        self, acquisition_function: str, model: SingleTaskGP, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if acquisition_function == "ExpectedImprovement":
            acqf = ExpectedImprovement(model=model, best_f=target.min())
        elif acquisition_function == "UpperConfidenceBound":
            beta = 10.0
            acqf = UpperConfidenceBound(model=model, beta=beta)
        elif acquisition_function == "qExpectedImprovement":
            acqf = qExpectedImprovement(model=model, best_f=target.min())
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
        new_x, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=self.q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        new_designs = [
            LSMDesign(x[0].item(), np.ceil(x[1].item()), Policy.Leveling) for x in new_x
        ]
        # the new cost is also negated here
        new_y = torch.tensor(
            [
                -self.cf.calc_cost(
                    design,
                    self.system,
                    self.workload.z0,
                    self.workload.z1,
                    self.workload.q,
                    self.workload.w,
                )
                for design in new_designs
            ]
        )
        return new_x, new_y

    def _update_best_designs(self, new_x: torch.Tensor, new_y: torch.Tensor) -> None:
        for x, y in zip(new_x, new_y):
            self.best_designs.append(
                (
                    LSMDesign(x[0].item(), np.ceil(x[1].item()), Policy.Leveling),
                    y.item(),
                )
            )

    def _print_best_designs(self) -> None:
        # self.best_designs.sort(key=lambda x: x[1])
        print("Best Designs Found:")
        # for design, cost in self.best_designs[:5]:
        for design, cost in self.best_designs:
            print(
                f"Design: h={design.h}, T={design.T}, Policy={design.policy}, Cost={cost}"
            )

    def _find_analytical_results(self):
        solver = ClassicSolver(self.config)
        nominal_design, nominal_solution = solver.get_nominal_design(
            system=self.system,
            z0=self.workload.z0,
            z1=self.workload.z1,
            q=self.workload.q,
            w=self.workload.w,
        )
        x = np.array([nominal_design.h, nominal_design.T])
        policy = nominal_design.policy
        cost = solver.nominal_objective(
            x,
            policy,
            self.system,
            self.workload.z0,
            self.workload.z1,
            self.workload.q,
            self.workload.w,
        )
        print("Cost for the nominal design using analytical solver: ", cost)
        print("Nominal Design suggested by analytical solver: ", nominal_design)


if __name__ == "__main__":
    config = Reader.read_config("endure.toml")

    log = logging.getLogger(config["log"]["name"])
    log.info("Initializing Bayesian Optimization Job")

    bayesian_optimizer = BayesianPipeline(config)
    bayesian_optimizer.run()
