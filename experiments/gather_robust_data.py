import logging
import os
import sqlite3

import toml
import torch
from tqdm import tqdm

from axe.lsm.cost import Cost
from axe.lsm.solver import ClassicSolver, KLSMSolver
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload
from axe.ltuner.data.schema import LTunerDataSchema
from axe.ltuner.model.builder import LTuneModelBuilder
from experiments.infra import workloads as predefined_workloads

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gather_robust_data")


class RobustEvaluationDB:
    def __init__(self, db_path: str, max_kap_levels: int = 20):
        self.db_path = db_path
        self.max_kap_levels = max_kap_levels
        self.con = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.con.cursor()
        # Environments Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS environments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                z0 REAL, z1 REAL, q REAL, w REAL,
                entry_size INTEGER, selectivity REAL, entries_per_page INTEGER,
                num_entries INTEGER, mem_budget REAL
            )
        """)
        # Nominal Designs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nominal_designs (
                id INTEGER,
                bits_per_elem REAL, size_ratio REAL, policy TEXT, cost REAL,
                FOREIGN KEY(id) REFERENCES environments(id)
            )
        """)
        # Robust Classic Designs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS robust_designs (
                id INTEGER,
                bits_per_elem REAL, size_ratio REAL, policy TEXT,
                rho REAL, cost REAL,
                FOREIGN KEY(id) REFERENCES environments(id)
            )
        """)
        # Robust Learned Designs Table
        kap_cols = ", ".join([f"kap{i} REAL" for i in range(self.max_kap_levels)])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS robust_learned (
                id INTEGER,
                bits_per_elem REAL, size_ratio REAL, policy TEXT,
                {kap_cols},
                rho REAL, cost REAL,
                FOREIGN KEY(id) REFERENCES environments(id)
            )
        """)
        self.con.commit()

    def insert_environment(self, w: Workload, s: System) -> int:
        cursor = self.con.cursor()
        cursor.execute(
            """
            INSERT INTO environments (z0, z1, q, w, entry_size, selectivity, entries_per_page, num_entries, mem_budget)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                w.z0,
                w.z1,
                w.q,
                w.w,
                s.entry_size,
                s.selectivity,
                s.entries_per_page,
                s.num_entries,
                s.mem_budget,
            ),
        )
        self.con.commit()
        return cursor.lastrowid

    def insert_nominal(self, env_id: int, design: LSMDesign, cost: float):
        cursor = self.con.cursor()
        cursor.execute(
            """
            INSERT INTO nominal_designs (id, bits_per_elem, size_ratio, policy, cost)
            VALUES (?, ?, ?, ?, ?)
        """,
            (env_id, design.bits_per_elem, design.size_ratio, str(design.policy), cost),
        )
        self.con.commit()

    def insert_robust_classic(
        self, env_id: int, design: LSMDesign, rho: float, cost: float
    ):
        cursor = self.con.cursor()
        cursor.execute(
            """
            INSERT INTO robust_designs (id, bits_per_elem, size_ratio, policy, rho, cost)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                env_id,
                design.bits_per_elem,
                design.size_ratio,
                str(design.policy),
                rho,
                cost,
            ),
        )
        self.con.commit()

    def insert_robust_learned(
        self, env_id: int, design: LSMDesign, rho: float, cost: float
    ):
        cursor = self.con.cursor()
        kap_vals = list(design.kapacity)
        # Pad with zeros if necessary
        kap_vals += [0.0] * (self.max_kap_levels - len(kap_vals))

        cols = (
            ["id", "bits_per_elem", "size_ratio", "policy"]
            + [f"kap{i}" for i in range(self.max_kap_levels)]
            + ["rho", "cost"]
        )
        placeholders = ", ".join(["?"] * len(cols))
        values = (
            [env_id, design.bits_per_elem, design.size_ratio, str(design.policy)]
            + kap_vals
            + [rho, cost]
        )

        cursor.execute(
            f"INSERT INTO robust_learned ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )
        self.con.commit()

    def get_existing_env_count(self) -> int:
        cursor = self.con.cursor()
        cursor.execute("SELECT COUNT(*) FROM environments")
        return cursor.fetchone()[0]


class RobustEvaluator:
    def __init__(self, model_path: str):
        self.cfg = toml.load(os.path.join(model_path, "axe.toml"))
        policy: Policy = getattr(Policy, self.cfg["lsm"]["policy"])
        self.bounds: LSMBounds = LSMBounds(**self.cfg["lsm"]["bounds"])
        self.schema = LTunerDataSchema(policy=policy, bounds=self.bounds, robust=True)
        self.model = LTuneModelBuilder(
            self.schema, **self.cfg["ltuner"]["model"]
        ).build(robust=True)

        model_data = torch.load(
            os.path.join(model_path, "best_model.model"),
            map_location="cuda:0",
            weights_only=True,
        )
        self.model.load_state_dict(model_data["model_state_dict"])
        self.model.eval()

        self.cost_fn = Cost(self.bounds.max_considered_levels)
        self.classic_solver = ClassicSolver(self.bounds)
        self.klsm_solver = KLSMSolver(
            self.bounds
        )  # Though user says we use classic for robust designs, learned tuner is KLSM

    def convert_output_to_design(
        self, output: torch.Tensor, x: torch.Tensor
    ) -> LSMDesign:
        # KapLSMRobustTuner forward returns concat([lagrangians, bits, t, k], dim=-1)
        # lagrangians: 2, bits: 1, t: capacity_range, k: num_kap * capacity_range
        out = output.flatten()
        # lagrangians = out[0:2]
        bits = out[2].item()

        cap_range = (
            self.bounds.size_ratio_range[1] - self.bounds.size_ratio_range[0] + 1
        )
        t_probs = out[3 : 3 + cap_range]
        size_ratio = torch.argmax(t_probs).item() + self.bounds.size_ratio_range[0]

        k_probs = out[3 + cap_range :].reshape(-1, cap_range)
        # We need to filter k based on max levels, but let's just take all and the cost function will handle it if we create a design correctly.
        # Actually KapLSMRobustTuner.forward already masks k.

        # Determine num levels from bits and size_ratio to correctly slice k
        # Or just take all non-trivial ones.
        # For simplicity, let's just argmax all k's.
        kapacity = tuple(torch.argmax(k_row).item() + 1 for k_row in k_probs)

        return LSMDesign(
            bits_per_elem=bits,
            size_ratio=size_ratio,
            policy=Policy.Kapacity,
            kapacity=kapacity,
        )

    def get_learned_design(
        self, workload: Workload, system: System, rho: float
    ) -> LSMDesign:
        # Construct input vector for schema
        # ["z0", "z1", "q", "w", "entries_per_page", "selectivity", "entry_size", "mem_budget", "num_entries", "rho"]
        feat = torch.tensor(
            [
                [
                    workload.z0,
                    workload.z1,
                    workload.q,
                    workload.w,
                    system.entries_per_page,
                    system.selectivity,
                    system.entry_size,
                    system.mem_budget,
                    system.num_entries,
                    rho,
                ]
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():
            out = self.model(feat)

        return self.convert_output_to_design(out, feat)


def run_evaluation(num_random_workloads: int = 10000):
    db_path = "data/robust_evaluation.db"
    model_path = "data/ltuner/models/robust_kaplsm_260213_1613"

    evaluator = RobustEvaluator(model_path)
    db = RobustEvaluationDB(
        db_path, max_kap_levels=evaluator.bounds.max_considered_levels
    )

    existing_count = db.get_existing_env_count()

    # 1. Generate/Get Environments
    envs_to_process = []

    if existing_count == 0:
        logger.info("Initializing environments with predefined workloads...")
        system = System()  # Nominal system
        for w in predefined_workloads:
            env_id = db.insert_environment(w, system)
            envs_to_process.append((env_id, w, system))

    # Generate more random workloads if needed
    current_count = db.get_existing_env_count()
    needed = max(0, (15 + num_random_workloads) - current_count)
    if needed > 0:
        logger.info(f"Generating {needed} random environments...")
        for _ in tqdm(range(needed), desc="Generating Envs"):
            w = evaluator.schema.gen.sample_workload()
            s = System()  # Stick to nominal system as per usual evaluation or sample?
            # User said "nominal tuner, robust classic tuner, and robust learned tuner"
            # Usually we vary workload.
            env_id = db.insert_environment(w, s)
            envs_to_process.append((env_id, w, s))

    # If we didn't just create them, we might need to fetch them to process?
    # For now, let's assume we process what we just added, or if none added, we might want to process all.
    # To support "continue to run", we should check which envs lack designs.

    cursor = db.con.cursor()
    cursor.execute("""
        SELECT e.id, e.z0, e.z1, e.q, e.w, e.entry_size, e.selectivity, e.entries_per_page, e.num_entries, e.mem_budget
        FROM environments e
        LEFT JOIN nominal_designs n ON e.id = n.id
        WHERE n.id IS NULL
    """)
    rows = cursor.fetchall()

    rhos = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    for row in tqdm(rows, desc="Evaluating"):
        env_id = row[0]
        workload = Workload(row[1], row[2], row[3], row[4])
        system = System(
            entry_size=row[5],
            selectivity=row[6],
            entries_per_page=row[7],
            num_entries=row[8],
            mem_budget=row[9],
        )

        # Nominal Design
        nom_design, _ = evaluator.classic_solver.get_nominal_design(system, workload)
        nom_cost = evaluator.cost_fn.calc_cost(nom_design, system, workload)
        db.insert_nominal(env_id, nom_design, nom_cost)

        for rho in rhos:
            # Robust Classic
            rob_design, _ = evaluator.classic_solver.get_robust_design(
                system, workload, rho
            )
            rob_cost = evaluator.cost_fn.calc_cost(rob_design, system, workload)
            db.insert_robust_classic(env_id, rob_design, rho, rob_cost)

            # Robust Learned
            lrn_design = evaluator.get_learned_design(workload, system, rho)
            lrn_cost = evaluator.cost_fn.calc_cost(lrn_design, system, workload)
            db.insert_robust_learned(env_id, lrn_design, rho, lrn_cost)


if __name__ == "__main__":
    import sys

    n_rand = 50000
    if len(sys.argv) > 1:
        n_rand = int(sys.argv[1])
    run_evaluation(n_rand)
