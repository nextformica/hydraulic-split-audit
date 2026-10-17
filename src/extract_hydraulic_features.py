#!/usr/bin/env python3
"""Extract auditable, cycle-level summary features from the UCI archive.

Feature extraction is deliberately separate from model fitting.  The resulting
CSV contains one row per 60-second cycle; scaling/selection must be fitted
inside each training fold by the modelling script.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np


SENSORS = [
    ("PS1", "PS1.txt"), ("PS2", "PS2.txt"), ("PS3", "PS3.txt"),
    ("PS4", "PS4.txt"), ("PS5", "PS5.txt"), ("PS6", "PS6.txt"),
    ("EPS1", "EPS1.txt"), ("FS1", "FS1.txt"), ("FS2", "FS2.txt"),
    ("TS1", "TS1.txt"), ("TS2", "TS2.txt"), ("TS3", "TS3.txt"),
    ("TS4", "TS4.txt"), ("VS1", "VS1.txt"), ("CE", "CE.txt"),
    ("CP", "CP.txt"), ("SE", "SE.txt"),
]
STAT_NAMES = ["mean", "std", "min", "max", "range", "q10", "median", "q90", "end_start_delta", "mean_abs_diff"]
LABELS = ["cooler", "valve", "pump_leak", "accumulator", "stable"]


def row_features(raw: bytes) -> list[float]:
    values = np.fromstring(raw.decode("ascii").strip(), sep=" ", dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("empty or non-finite signal row")
    differences = np.diff(values)
    q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.ptp(values)),
        float(q10),
        float(median),
        float(q90),
        float(values[-1] - values[0]),
        float(np.mean(np.abs(differences))) if differences.size else 0.0,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("raw/hydraulic_uci_probe/uci_hydraulic_official.zip"))
    parser.add_argument("--registry", type=Path, default=Path("results/hydraulic_uci_cycles.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/hydraulic_uci_cycle_features.csv"))
    args = parser.parse_args()
    labels = list(csv.DictReader(args.registry.open(encoding="utf-8-sig", newline="")))
    if len(labels) != 2205:
        raise ValueError(f"registry contains {len(labels)} cycles, expected 2205")
    feature_names = [f"{sensor}_{stat}" for sensor, _ in SENSORS for stat in STAT_NAMES]
    rows: list[dict[str, object]] = [
        {"cycle": int(row["cycle"]), "block_id": int(row["block_id"]), **{name: int(row[name]) for name in LABELS}}
        for row in labels
    ]
    with zipfile.ZipFile(args.zip.resolve()) as zf:
        for sensor, member in SENSORS:
            with zf.open(member, "r") as stream:
                for index, raw in enumerate(stream):
                    if index >= len(rows):
                        raise ValueError(f"{member} has more than 2205 rows")
                    values = row_features(raw)
                    rows[index].update(dict(zip((f"{sensor}_{stat}" for stat in STAT_NAMES), values)))
            if not all(name in rows[-1] for name in (f"{sensor}_{stat}" for stat in STAT_NAMES)):
                raise ValueError(f"{member} has fewer than 2205 rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cycle", "block_id", *LABELS, *feature_names]
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print({"cycles": len(rows), "features": len(feature_names), "output": str(args.output.resolve())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
