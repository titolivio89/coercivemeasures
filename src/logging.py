"""Small logging helper for experiments (lightweight csv-based logging)."""
import os
import csv
from typing import Dict


def log_experiment_row(path: str, row: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = list(row.keys())
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
