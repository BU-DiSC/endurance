import sqlite3
from endure.data.io import Reader
from endure.lsm.types import LSMDesign, Policy


def initialize_database(db_path='cost_log.db'):
    connector = sqlite3.connect(db_path)
    cursor = connector.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            empty_reads REAL,
            non_empty_reads REAL,
            range_queries REAL,
            writes REAL,
            max_bits_per_element REAL,
            physical_entries_per_page INT,
            range_selectivity REAL,
            entries_per_page INT,
            total_elements INT,
            read_write_asymmetry REAL,
            iterations INT,
            sample_size INT,
            acquisition_function TEXT
        );''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS design_costs (
            idx INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            bits_per_element REAL,
            size_ratio INTEGER,
            policy TEXT,
            Q INTEGER,
            Y INTEGER,
            Z INTEGER,
            cost REAL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS run_details (
            run_id INTEGER PRIMARY KEY,
            duration_secs INTEGER,
            analytical_cost REAL,
            bayesian_cost REAL,
            analytical_h REAL,
            analytical_T INTEGER,
            analytical_policy TEXT,
            analytical_Q INTEGER,
            analytical_Y INTEGER,
            analytical_Z INTEGER,
            bayesian_h REAL,
            bayesian_T INTEGER,
            bayesian_policy TEXT,
            bayesian_Q INTEGER,
            bayesian_Y INTEGER,
            bayesian_Z INTEGER,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );''')
    connector.commit()
    return connector


def log_new_run(connector, system, workload, iterations, sample, acqf):
    cursor = connector.cursor()
    cursor.execute('INSERT INTO runs (empty_reads, non_empty_reads, range_queries, writes, '
                   'max_bits_per_element, physical_entries_per_page, range_selectivity, '
                   'entries_per_page, total_elements, read_write_asymmetry, iterations, sample_size, '
                   'acquisition_function) '
                   'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   (workload.z0,  workload.z1, workload.q, workload.w, system.H, system.E, system.s, system.B,
                    system.N, system.phi, iterations, sample, acqf))
    connector.commit()
    return cursor.lastrowid


def log_design_cost(connector, run_id, design, cost):
    cursor = connector.cursor()
    policy = design.policy
    cursor.execute('INSERT INTO design_costs (run_id, bits_per_element, size_ratio, policy, Q, Y, Z, cost) '
                   'VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (run_id, design.h, design.T, policy.name, design.Q, design.Y,
                                                       design.Z, cost))
    connector.commit()


def log_run_details(connector, run_id, duration, analytical_cost, bayesian_cost, analytical_design, bayesian_design):
    cursor = connector.cursor()
    analytical_policy = analytical_design.policy
    print(analytical_policy)
    bayesian_policy = bayesian_design.policy
    if bayesian_policy == 0.0:
        bo_policy = Policy.Leveling
    else:
        bo_policy = Policy.Tiering
    cursor.execute('''
        INSERT INTO run_details (run_id, duration_secs, analytical_cost, bayesian_cost, analytical_h, analytical_T, 
        analytical_policy, analytical_Q, analytical_Y, analytical_Z, bayesian_h, bayesian_T, bayesian_policy, bayesian_Q, 
        bayesian_Y, bayesian_Z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (run_id, duration, analytical_cost, bayesian_cost, analytical_design.h, analytical_design.T,
                    analytical_policy.name, analytical_design.Q, analytical_design.Y, analytical_design.Z,
                    bayesian_design.h, bayesian_design.T, bo_policy.name, bayesian_design.Q,
                    bayesian_design.Y, bayesian_design.Z))
    connector.commit()


if __name__ == "__main__":
    config = Reader.read_config("endure.toml")
    conn = initialize_database()
    conn.close()
