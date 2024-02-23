import sqlite3
from endure.data.io import Reader


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
            policy INTEGER,
            Q INTEGER,
            Y INTEGER,
            Z INTEGER,
            cost REAL,
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

    policy_value = design.policy.value if hasattr(design.policy, 'value') else design.policy
    cursor.execute('INSERT INTO design_costs (run_id, bits_per_element, size_ratio, policy, Q, Y, Z, cost) '
                   'VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (run_id, design.h, design.T, policy_value, design.Q, design.Y,
                                                       design.Z, cost))
    connector.commit()


if __name__ == "__main__":
    config = Reader.read_config("endure.toml")
    conn = initialize_database()
    conn.close()
