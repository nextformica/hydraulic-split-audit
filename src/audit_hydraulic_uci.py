#!/usr/bin/env python3
"""Audit the official UCI hydraulic condition-monitoring archive.

The archive is kept read-only.  This script streams each text member so that
the full-width pressure/power files can be checked without loading the whole
dataset into memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Iterable


SIGNAL_COLUMNS = {
    "PS1.txt": (2205, 6000, 100.0),
    "PS2.txt": (2205, 6000, 100.0),
    "PS3.txt": (2205, 6000, 100.0),
    "PS4.txt": (2205, 6000, 100.0),
    "PS5.txt": (2205, 6000, 100.0),
    "PS6.txt": (2205, 6000, 100.0),
    "EPS1.txt": (2205, 6000, 100.0),
    "FS1.txt": (2205, 600, 10.0),
    "FS2.txt": (2205, 600, 10.0),
    "TS1.txt": (2205, 60, 1.0),
    "TS2.txt": (2205, 60, 1.0),
    "TS3.txt": (2205, 60, 1.0),
    "TS4.txt": (2205, 60, 1.0),
    "VS1.txt": (2205, 60, 1.0),
    "CE.txt": (2205, 60, 1.0),
    "CP.txt": (2205, 60, 1.0),
    "SE.txt": (2205, 60, 1.0),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_numbers(raw: bytes) -> list[float]:
    text = raw.decode("utf-8", errors="strict").strip()
    if not text:
        return []
    return [float(token) for token in text.replace(",", " ").split()]


def audit_member(zf: zipfile.ZipFile, name: str, expected_rows: int, expected_cols: int, hz: float) -> dict:
    rows = 0
    nonempty_rows = 0
    malformed_rows = 0
    blank_rows = 0
    bad_examples: list[dict] = []
    finite = 0
    nonfinite = 0
    minimum = math.inf
    maximum = -math.inf
    total = 0.0
    total_sq = 0.0
    min_cols = math.inf
    max_cols = 0
    with zf.open(name, "r") as stream:
        for raw in stream:
            rows += 1
            try:
                values = parse_numbers(raw)
            except (UnicodeDecodeError, ValueError) as exc:
                malformed_rows += 1
                if len(bad_examples) < 5:
                    bad_examples.append({"row": rows, "error": str(exc)})
                continue
            if not values:
                blank_rows += 1
                continue
            nonempty_rows += 1
            ncols = len(values)
            min_cols = min(min_cols, ncols)
            max_cols = max(max_cols, ncols)
            if ncols != expected_cols and len(bad_examples) < 5:
                bad_examples.append({"row": rows, "columns": ncols, "expected_columns": expected_cols})
            for value in values:
                if math.isfinite(value):
                    finite += 1
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += value
                    total_sq += value * value
                else:
                    nonfinite += 1
    count = finite
    mean = total / count if count else None
    variance = max(total_sq / count - mean * mean, 0.0) if count and mean is not None else None
    return {
        "file": name,
        "sampling_hz": hz,
        "expected_rows": expected_rows,
        "observed_rows": rows,
        "expected_columns": expected_cols,
        "nonempty_rows": nonempty_rows,
        "min_columns": None if math.isinf(min_cols) else int(min_cols),
        "max_columns": max_cols,
        "finite_values": finite,
        "nonfinite_values": nonfinite,
        "malformed_rows": malformed_rows,
        "blank_rows": blank_rows,
        "min": None if math.isinf(minimum) else minimum,
        "max": None if maximum == -math.inf else maximum,
        "mean": mean,
        "std_population": math.sqrt(variance) if variance is not None else None,
        "shape_ok": rows == expected_rows and nonempty_rows == expected_rows and min_cols == expected_cols and max_cols == expected_cols and malformed_rows == 0 and finite == expected_rows * expected_cols,
        "finite_ok": nonfinite == 0,
        "bad_examples": bad_examples,
    }


def audit_profile(zf: zipfile.ZipFile, expected_rows: int) -> dict:
    rows: list[list[int]] = []
    malformed: list[dict] = []
    with zf.open("profile.txt", "r") as stream:
        for row_no, raw in enumerate(stream, 1):
            try:
                values = [int(token) for token in raw.decode("utf-8").strip().split()]
            except (UnicodeDecodeError, ValueError) as exc:
                malformed.append({"row": row_no, "error": str(exc)})
                continue
            if values:
                rows.append(values)
    widths = sorted({len(row) for row in rows})
    return {
        "file": "profile.txt",
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "row_widths": widths,
        "malformed_rows": malformed[:5],
        "shape_ok": len(rows) == expected_rows and widths == [5] and not malformed,
        "label_values": [sorted({row[i] for row in rows}) for i in range(5)] if rows and widths == [5] else [],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("raw/hydraulic_uci_probe/uci_hydraulic_official.zip"))
    parser.add_argument("--json", type=Path, default=Path("results/hydraulic_uci_full_audit.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/hydraulic_uci_signal_audit.csv"))
    args = parser.parse_args()
    zip_path = args.zip.resolve()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = {entry.filename for entry in zf.infolist()}
        missing = sorted((set(SIGNAL_COLUMNS) | {"profile.txt"}) - names)
        if missing:
            raise RuntimeError(f"archive members missing: {missing}")
        signals = [audit_member(zf, name, *SIGNAL_COLUMNS[name]) for name in SIGNAL_COLUMNS]
        profile = audit_profile(zf, 2205)
    result = {
        "archive": str(zip_path),
        "sha256": digest,
        "member_count": len(names),
        "signal_count": len(signals),
        "signals": signals,
        "profile": {k: v for k, v in profile.items() if k != "rows"},
        "overall_shape_ok": all(row["shape_ok"] for row in signals) and profile["shape_ok"],
        "overall_finite_ok": all(row["finite_ok"] for row in signals),
    }
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["file", "sampling_hz", "expected_rows", "observed_rows", "nonempty_rows", "expected_columns", "min_columns", "max_columns", "finite_values", "nonfinite_values", "malformed_rows", "blank_rows", "min", "max", "mean", "std_population", "shape_ok", "finite_ok"]
    with args.csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in signals)
    print(json.dumps({k: result[k] for k in ["sha256", "member_count", "signal_count", "overall_shape_ok", "overall_finite_ok"]}, ensure_ascii=False))
    return 0 if result["overall_shape_ok"] and result["overall_finite_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
