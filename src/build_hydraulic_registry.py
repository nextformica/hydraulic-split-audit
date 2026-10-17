#!/usr/bin/env python3
"""Register UCI hydraulic cycles and conservative contiguous operating blocks."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["cooler", "valve", "pump_leak", "accumulator", "stable"]


def read_profile(zip_path: Path) -> list[list[int]]:
    with zipfile.ZipFile(zip_path) as zf, zf.open("profile.txt") as stream:
        rows = [[int(x) for x in line.decode("utf-8").split()] for line in stream if line.strip()]
    if len(rows) != 2205 or any(len(row) != 5 for row in rows):
        raise ValueError(f"profile shape is not 2205x5: {len(rows)} rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("raw/hydraulic_uci_probe/uci_hydraulic_official.zip"))
    parser.add_argument("--cycles", type=Path, default=Path("results/hydraulic_uci_cycles.csv"))
    parser.add_argument("--blocks", type=Path, default=Path("results/hydraulic_uci_blocks.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/hydraulic_uci_registry_summary.json"))
    args = parser.parse_args()
    rows = read_profile(args.zip.resolve())
    cycles: list[dict] = []
    block_id = -1
    previous = None
    block_start = 0
    blocks: list[dict] = []
    for index, values in enumerate(rows):
        state = tuple(values)
        if state != previous:
            if previous is not None:
                blocks[-1]["end_cycle"] = index - 1
                blocks[-1]["length"] = index - block_start
            block_id += 1
            block_start = index
            blocks.append({"block_id": block_id, "start_cycle": index, "state": list(state)})
        cycle = {"cycle": index, "block_id": block_id}
        cycle.update(dict(zip(LABELS, values)))
        cycles.append(cycle)
        previous = state
    blocks[-1]["end_cycle"] = len(rows) - 1
    blocks[-1]["length"] = len(rows) - block_start
    for block in blocks:
        block.update(dict(zip(LABELS, block.pop("state"))))

    args.cycles.parent.mkdir(parents=True, exist_ok=True)
    args.blocks.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    cycle_fields = ["cycle", "block_id", *LABELS]
    block_fields = ["block_id", "start_cycle", "end_cycle", "length", *LABELS]
    with args.cycles.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cycle_fields)
        writer.writeheader()
        writer.writerows(cycles)
    with args.blocks.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=block_fields)
        writer.writeheader()
        writer.writerows(blocks)

    summary: dict = {
        "cycles": len(cycles),
        "blocks": len(blocks),
        "block_length_counts": dict(sorted(Counter(b["length"] for b in blocks).items())),
        "unique_label_vectors": len({tuple(row[name] for name in LABELS) for row in cycles}),
        "labels": {},
        "tasks": {},
    }
    for label in LABELS:
        summary["labels"][label] = dict(sorted(Counter(row[label] for row in cycles).items(), key=lambda item: item[0]))
    for target in ["accumulator", "pump_leak", "cooler", "valve"]:
        by_block: dict[int, set[int]] = defaultdict(set)
        for row in cycles:
            by_block[row["block_id"]].add(row[target])
        block_class_counts = Counter(next(iter(values)) for values in by_block.values() if len(values) == 1)
        mixed_blocks = sum(1 for values in by_block.values() if len(values) > 1)
        summary["tasks"][target] = {
            "classes": sorted(summary["labels"][target]),
            "blocks_with_single_class": dict(sorted(block_class_counts.items())),
            "mixed_label_blocks": mixed_blocks,
            "usable_group_count": len(by_block),
        }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cycles": len(cycles), "blocks": len(blocks), "unique_label_vectors": summary["unique_label_vectors"], "tasks": summary["tasks"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
