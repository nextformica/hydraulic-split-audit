#!/usr/bin/env python3
"""Nested grouped model comparison for the hydraulic cycle features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


LABELS = ["cooler", "valve", "pump_leak", "accumulator", "stable"]
COMPONENT_LABELS = ["cooler", "valve", "pump_leak", "accumulator"]
SENSOR_SETS = {
    "all": lambda col: any(col.startswith(prefix + "_") for prefix in ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]),
    "pressure_power": lambda col: any(col.startswith(prefix + "_") for prefix in ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]),
    "pressure_flow": lambda col: any(col.startswith(prefix + "_") for prefix in ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "FS1", "FS2"]),
    "low_rate": lambda col: any(col.startswith(prefix + "_") for prefix in ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]),
    "physical": lambda col: any(col.startswith(prefix + "_") for prefix in ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1"]),
}


def read_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError("feature CSV is empty")
    fields = list(records[0])
    data = {field: np.array([float(row[field]) for row in records], dtype=float) for field in fields}
    return fields, data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(X, y, groups))
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups))


def candidates() -> list[tuple[str, str, object]]:
    models: list[tuple[str, str, object]] = []
    for c in [0.1, 1.0, 10.0]:
        models.append(("logistic", f"C={c}", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, max_iter=2500, class_weight=None, random_state=20261013))])))
    for c in [1.0, 10.0]:
        models.append(("rbf_svm", f"C={c}", Pipeline([("scale", StandardScaler()), ("model", SVC(C=c, kernel="rbf", gamma="scale"))])))
    models.append(("random_forest", "n=300", RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=20261013, n_jobs=-1)))
    return models


def grouped_accuracy(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> float:
    block_true: list[float] = []
    block_pred: list[float] = []
    for block in sorted(set(group_ids.astype(int))):
        mask = group_ids == block
        block_true.append(Counter(y_true[mask]).most_common(1)[0][0])
        block_pred.append(Counter(y_pred[mask]).most_common(1)[0][0])
    return float(accuracy_score(block_true, block_pred))


def grouped_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> float:
    group_true: list[float] = []
    group_pred: list[float] = []
    for group in sorted(set(group_ids.astype(int))):
        mask = group_ids == group
        group_true.append(Counter(y_true[mask]).most_common(1)[0][0])
        group_pred.append(Counter(y_pred[mask]).most_common(1)[0][0])
    return float(balanced_accuracy_score(group_true, group_pred))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("results/hydraulic_uci_cycle_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/hydraulic_uci_nested_grouped_results.json"))
    parser.add_argument("--task", choices=["accumulator", "pump_leak"], default="accumulator")
    parser.add_argument("--sensor-set", choices=sorted(SENSOR_SETS), default="all")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--stable-flag", choices=["all", "0", "1"], default="all", help="restrict cycles by the profile stable flag")
    parser.add_argument("--grouping", choices=["random", "block", "state4", "state5"], default="state4", help="random = cycle-level stratified baseline; block = contiguous run; state4 = exact four-component vector; state5 also includes stable flag")
    args = parser.parse_args()
    fields, data = read_csv(args.features.resolve())
    feature_names = [field for field in fields if SENSOR_SETS[args.sensor_set](field)]
    if not feature_names:
        raise ValueError("no feature columns selected")
    X = np.column_stack([data[field] for field in feature_names])
    y = data[args.task].astype(int)
    component_state_keys = ["|".join(str(int(data[label][i])) for label in COMPONENT_LABELS) for i in range(len(y))]
    _, component_state_ids = np.unique(np.asarray(component_state_keys), return_inverse=True)
    full_state_keys = ["|".join(str(int(data[label][i])) for label in LABELS) for i in range(len(y))]
    _, full_state_ids = np.unique(np.asarray(full_state_keys), return_inverse=True)
    if args.grouping == "random":
        groups = np.arange(len(y), dtype=int)
    elif args.grouping == "block":
        groups = data["block_id"].astype(int)
    elif args.grouping == "state4":
        groups = component_state_ids
    else:
        groups = full_state_ids
    if args.stable_flag != "all":
        keep = data["stable"].astype(int) == int(args.stable_flag)
        X, y, groups = X[keep], y[keep], groups[keep]
        contiguous_groups = data["block_id"].astype(int)[keep]
    else:
        contiguous_groups = data["block_id"].astype(int)
    if len(set(groups)) < args.outer_folds:
        raise ValueError(f"only {len(set(groups))} groups remain for {args.outer_folds} folds")
    outer = splits(X, y, groups, args.outer_folds, 20261013)
    all_fold_results: list[dict] = []
    oof = np.full(len(y), -1, dtype=int)
    oof_fold = np.full(len(y), -1, dtype=int)
    oof_model: list[str] = [""] * len(y)
    for fold, (train_idx, test_idx) in enumerate(outer, 1):
        inner_n = 3 if len(set(groups[train_idx])) >= 3 else 2
        inner = splits(X[train_idx], y[train_idx], groups[train_idx], inner_n, 20261013 + fold)
        selection: list[dict] = []
        for name, params, model in candidates():
            scores = []
            for inner_train_rel, inner_valid_rel in inner:
                model.fit(X[train_idx][inner_train_rel], y[train_idx][inner_train_rel])
                pred = model.predict(X[train_idx][inner_valid_rel])
                scores.append(float(balanced_accuracy_score(y[train_idx][inner_valid_rel], pred)))
            selection.append({"model": name, "params": params, "inner_balanced_accuracy": float(np.mean(scores)), "inner_fold_scores": scores})
        best = max(selection, key=lambda row: row["inner_balanced_accuracy"])
        best_model = next(model for name, params, model in candidates() if name == best["model"] and params == best["params"])
        best_model.fit(X[train_idx], y[train_idx])
        pred = best_model.predict(X[test_idx])
        oof[test_idx] = pred
        oof_fold[test_idx] = fold
        for index in test_idx:
            oof_model[index] = f"{best['model']}:{best['params']}"
        fold_result = {
            "fold": fold,
            "train_cycles": int(len(train_idx)),
            "test_cycles": int(len(test_idx)),
            "train_groups": int(len(set(groups[train_idx]))),
            "test_groups": int(len(set(groups[test_idx]))),
            "train_contiguous_blocks": int(len(set(contiguous_groups[train_idx]))),
            "test_contiguous_blocks": int(len(set(contiguous_groups[test_idx]))),
            "train_class_counts": {str(int(key)): int(value) for key, value in sorted(Counter(y[train_idx].astype(int)).items())},
            "test_class_counts": {str(int(key)): int(value) for key, value in sorted(Counter(y[test_idx].astype(int)).items())},
            "selected_model": best,
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
            "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
            "evaluation_group_accuracy": grouped_accuracy(y[test_idx], pred, groups[test_idx]),
            "evaluation_group_balanced_accuracy": grouped_balanced_accuracy(y[test_idx], pred, groups[test_idx]),
            "contiguous_block_accuracy": grouped_accuracy(y[test_idx], pred, contiguous_groups[test_idx]),
        }
        all_fold_results.append(fold_result)
    aggregate = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "evaluation_group_accuracy", "evaluation_group_balanced_accuracy", "contiguous_block_accuracy"]:
        values = [row[metric] for row in all_fold_results]
        aggregate[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "folds": values}
    aggregate["pooled_oof"] = {
        "accuracy": float(accuracy_score(y, oof)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "macro_f1": float(f1_score(y, oof, average="macro")),
        "evaluation_group_accuracy": grouped_accuracy(y, oof, groups),
        "evaluation_group_balanced_accuracy": grouped_balanced_accuracy(y, oof, groups),
        "contiguous_block_accuracy": grouped_accuracy(y, oof, contiguous_groups),
    }
    result = {
        "task": args.task,
        "sensor_set": args.sensor_set,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "group_definition": {
            "random": "cycle-level stratified baseline; adjacent or repeated states may cross folds",
            "block": "consecutive identical five-label vectors; same block never crosses folds",
            "state4": "exact four-component vector (cooler, valve, pump leak, accumulator); stable is retained for stratified reporting, not grouping",
            "state5": "exact five-label vector including stable flag; retained only as a sensitivity analysis",
        }[args.grouping],
        "grouping": args.grouping,
        "splitter": "StratifiedGroupKFold" if StratifiedGroupKFold is not None else "GroupKFold",
        "outer_folds": args.outer_folds,
        "stable_flag_filter": args.stable_flag,
        "random_seed": 20261013,
        "data_sha256": file_sha256(args.features.resolve()),
        "n_samples_after_filter": int(len(y)),
        "oof_complete": bool(np.all(oof >= 0)),
        "folds": all_fold_results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    oof_path = args.output.with_name(args.output.stem + "_oof.csv")
    original_indices = np.arange(len(data["cycle"]))
    if args.stable_flag != "all":
        original_indices = original_indices[keep]
    with oof_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cycle", "block_id", "state4_id", "stable", "y_true", "y_pred", "fold", "model"])
        for local_index, original_index in enumerate(original_indices):
            writer.writerow([int(data["cycle"][original_index]), int(data["block_id"][original_index]), int(component_state_ids[original_index]), int(data["stable"][original_index]), int(y[local_index]), int(oof[local_index]), int(oof_fold[local_index]), oof_model[local_index]])
    print(json.dumps({"task": args.task, "sensor_set": args.sensor_set, "feature_count": len(feature_names), "aggregate": aggregate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
