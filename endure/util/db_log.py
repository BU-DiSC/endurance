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
            k1 REAL, k2 REAL, k3 REAL, k4 REAL, k5 REAL,
            k6 REAL, k7 REAL, k8 REAL, k9 REAL, k10 REAL,
            k11 REAL, k12 REAL, k13 REAL, k14 REAL, k15 REAL,
            k16 REAL, k17 REAL, k18 REAL, k19 REAL, k20 REAL,
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS run_details_k_values (
            idx INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            k1 REAL, k2 REAL, k3 REAL, k4 REAL, k5 REAL,
            k6 REAL, k7 REAL, k8 REAL, k9 REAL, k10 REAL,
            k11 REAL, k12 REAL, k13 REAL, k14 REAL, k15 REAL,
            k16 REAL, k17 REAL, k18 REAL, k19 REAL, k20 REAL,
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
    k_values = design.K + [None] * (20 - len(design.K))  # TODO replace this with the max_levels
    sql_command = ('INSERT INTO design_costs (run_id, bits_per_element, size_ratio, policy, Q, Y, Z, cost, ' +
                   ', '.join([f'k{i+1}' for i in range(20)]) + ') ' + 'VALUES (?, ?, ?, ?, ?, ?, ?, ?'', ' +
                   ', '.join(['?']*20) + ')')
    cursor.execute(sql_command, (run_id, design.h, design.T, policy.name, design.Q, design.Y, design.Z, cost) +
                   tuple(k_values))
    connector.commit()


def log_run_details(connector, run_id, duration, analytical_cost, bayesian_cost, analytical_design, bayesian_design):
    cursor = connector.cursor()
    analytical_policy = analytical_design.policy
    print(analytical_policy)
    bayesian_policy = bayesian_design.policy
    cursor.execute('''
        INSERT INTO run_details (run_id, duration_secs, analytical_cost, bayesian_cost, analytical_h, analytical_T, 
        analytical_policy, analytical_Q, analytical_Y, analytical_Z, bayesian_h, bayesian_T, bayesian_policy, bayesian_Q, 
        bayesian_Y, bayesian_Z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (run_id, duration, analytical_cost, bayesian_cost, analytical_design.h, analytical_design.T,
                    analytical_policy.name, analytical_design.Q, analytical_design.Y, analytical_design.Z,
                    bayesian_design.h, bayesian_design.T, bayesian_policy.name, bayesian_design.Q,
                    bayesian_design.Y, bayesian_design.Z))
    if bayesian_policy == Policy.KHybrid:
        k_values = bayesian_design.K + [None] * (20 - len(bayesian_design.K))
        cursor.execute('''
            INSERT INTO run_details_k_values (run_id, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, 
            k16, k17, k18, k19, k20) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (run_id,) + tuple(k_values))
    connector.commit()


if __name__ == "__main__":
    config = Reader.read_config("endure.toml")
    conn = initialize_database()
    conn.close()
