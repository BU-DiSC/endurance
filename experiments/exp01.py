import logging
import pandas as pd
from data.dataio import Writer
from lsm.lsmtype import LSMTree, LSMSystem
from solver.nominalk import NominalQFixedTuning
from solver.nominal import NominalTierLevelTuning


class Exp01:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.log = logging.getLogger("endure")

    def get_endurek_trees(self) -> list[dict]:
        self.log.info("Calculating EndureK LSM Tree Tunings...")

        system = LSMSystem(**self.config["system"])
        default_tree = LSMTree(system)

        problem = NominalQFixedTuning(system)
        trees = []
        for id in self.config["exp_config"]["Exp01"]["wl_ids"]:
            wl = self.config["inputs"]["workloads"][id]
            self.log.debug(
                f'Workload: {wl["id"]}: {{z0: {wl["z0"]}, z1: {wl["z1"]}, q: {wl["q"]}, w: {wl["w"]}}}'
            )
            tree = problem.get_nominal_design(default_tree, wl)
            tree = tree.as_dict()
            tree.update(wl)
            trees.append(tree)
            self.log.debug(
                f'EndureK: h={tree["h"]:.2f}, T={tree["T"]:.2f}, Q={tree["Q"]:.2f}'
            )

        return trees

    def get_endure2_trees(self) -> list[dict]:
        self.log.info("Calculating EndureK LSM Tree Tunings...")

        system = LSMSystem(**self.config["system"])
        default_tree = LSMTree(system)

        problem = NominalTierLevelTuning(system)
        trees = []
        for id in self.config["exp_config"]["Exp01"]["wl_ids"]:
            wl = self.config["inputs"]["workloads"][id]
            self.log.debug(
                f'Workload: {wl["id"]}: {{z0: {wl["z0"]}, z1: {wl["z1"]}, q: {wl["q"]}, w: {wl["w"]}}}'
            )
            tree = problem.get_nominal_design(default_tree, wl)
            tree = tree.as_dict()
            tree.update(wl)
            trees.append(tree)
            self.log.debug(
                f'EndureK: h={tree["h"]:.2f}, T={tree["T"]:.2f}, policy={tree["policy"]}'
            )

        return trees

    def run(self) -> None:
        self.log.info("Experiment 01")
        writer = Writer(self.config)

        endurek_trees = pd.DataFrame(self.get_endurek_trees())
        endure2_trees = pd.DataFrame(self.get_endure2_trees())

        writer.export_csv_file(endurek_trees, "exp01_endurek_trees.csv")
        writer.export_csv_file(endure2_trees, "exp01_endure2_trees.csv")
