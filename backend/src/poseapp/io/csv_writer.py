# src/poseapp/io/csv_writer.py
import csv  # built-in CSV handling module
from typing import Iterable, Dict  # type hints for iterable collections and dict rows

def write_angles_csv(path: str, rows: Iterable[Dict]):  # write list/dict of angle records to CSV
    with open(path, "w", newline="", encoding="utf-8") as f:  # open CSV file for writing (UTF-8, no extra newlines)
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows)).keys()))  # create writer using keys from first dict
        w.writeheader()  # write column headers once
        for r in rows:  # iterate through all rows
            w.writerow(r)  # write each dictionary as a CSV line
