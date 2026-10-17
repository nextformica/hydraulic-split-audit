#!/usr/bin/env python3
"""Nested state4-grouped regression for the ordered accumulator pressure label.

The UCI documentation describes component values as degradation levels rather
than mutually exclusive fault events.  This script therefore treats accumulator
pressure as an ordered numeric target and keeps exact-classification analysis as
a secondary baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


LABELS = ["cooler", "valve", "pump_leak", "accumulator", "stable"]
COMPONENT_LABELS = LABELS[:4]
SENSOR_SETS = {
    "pressure_power": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"],
    "all": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"],
    "physical": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1"],
}


def read_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty feature table")
    fields = list(rows[0])
    return fields, {field: np.array([float(row[field]) for row in rows]) for field in fields}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def splitter(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int, method: str):
    # Each state4 group has one target level, but group-level stratification is
    # still useful here to avoid the severe level imbalance produced by the
    # order-preserving GroupKFold split. The target is passed as an integer
    # stratum; it is never used as a feature.
    if method == "stratified" and StratifiedGroupKFold is not None:
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y=y.astype(int), groups=groups))
    return list(GroupKFold(n_splits=n_splits).split(X, groups=groups))


def candidates():
    models = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        models.append(("ridge", f"alpha={alpha}", Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])))
    for c in [1.0, 10.0, 100.0]:
        models.append(("svr_rbf", f"C={c}", Pipeline([("scale", StandardScaler()), ("model", SVR(C=c, kernel="rbf", gamma="scale", epsilon=0.1))])))
    models.append(("random_forest", "n=300", RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=20261013, n_jobs=-1)))
    return models


def nearest_level(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return levels[np.argmin(np.abs(values[:, None] - levels[None, :]), axis=1)]


def metrics(y: np.ndarray, pred: np.ndarray, levels: np.ndarray) -> dict:
    nearest = nearest_level(pred, levels)
    ordinal_distance = np.abs(np.searchsorted(levels, nearest) - np.searchsorted(levels, y))
    return {
        "mae_bar": float(mean_absolute_error(y, pred)),
        "rmse_bar": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "nearest_level_accuracy": float(np.mean(nearest == y)),
        "within_one_level_accuracy": float(np.mean(ordinal_distance <= 1)),
        "mean_ordinal_distance": float(np.mean(ordinal_distance)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("results/hydraulic_uci_cycle_features.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-set", choices=sorted(SENSOR_SETS), default="pressure_power")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--stable-flag", choices=["all", "0", "1"], default="all")
    parser.add_argument("--splitter", choices=["stratified", "group"], default="stratified", help="outer and inner group split protocol")
    args = parser.parse_args()
    fields, data = read_csv(args.features.resolve())
    prefixes = SENSOR_SETS[args.sensor_set]
    feature_names = [field for field in fields if any(field.startswith(prefix + "_") for prefix in prefixes)]
    X = np.column_stack([data[field] for field in feature_names])
    y = data["accumulator"].astype(float)
    state_keys = ["|".join(str(int(data[label][i])) for label in COMPONENT_LABELS) for i in range(len(y))]
    _, groups = np.unique(np.asarray(state_keys), return_inverse=True)
    block_ids = data["block_id"].astype(int)
    if args.stable_flag != "all":
        keep = data["stable"].astype(int) == int(args.stable_flag)
        X, y, groups, block_ids = X[keep], y[keep], groups[keep], block_ids[keep]
    levels = np.array(sorted(set(y.astype(int))), dtype=float)
    outer = splitter(X, y, groups, args.outer_folds, 20261013, args.splitter)
    oof = np.full(len(y), np.nan)
    oof_fold = np.full(len(y), -1)
    oof_model = [""] * len(y)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(outer, 1):
        inner = splitter(X[train_idx], y[train_idx], groups[train_idx], 3, 20261013 + fold, args.splitter)
        selections = []
        for name, params, model in candidates():
            scores = []
            for inner_train, inner_valid in inner:
                model.fit(X[train_idx][inner_train], y[train_idx][inner_train])
                pred = model.predict(X[train_idx][inner_valid])
                scores.append(metrics(y[train_idx][inner_valid], pred, levels)["mae_bar"])
            selections.append({"model": name, "params": params, "inner_mae_bar": float(np.mean(scores)), "inner_fold_scores": scores})
        best = min(selections, key=lambda row: row["inner_mae_bar"])
        model = next(model for name, params, model in candidates() if name == best["model"] and params == best["params"])
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        oof[test_idx] = pred
        oof_fold[test_idx] = fold
        for i in test_idx:
            oof_model[i] = f"{best['model']}:{best['params']}"
        fold_rows.append({
            "fold": fold,
            "train_cycles": int(len(train_idx)),
            "test_cycles": int(len(test_idx)),
            "train_groups": int(len(set(groups[train_idx]))),
            "test_groups": int(len(set(groups[test_idx]))),
            "selected_model": best,
            "metrics": metrics(y[test_idx], pred, levels),
        })
    pooled = metrics(y, oof, levels)
    result = {
        "task": "accumulator_ordered_pressure_regression",
        "sensor_set": args.sensor_set,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_levels_bar": levels.astype(int).tolist(),
        "grouping": "state4",
        "group_definition": "exact four-component vector; stable is retained only for filtering/stratified reporting",
        "splitter": "StratifiedGroupKFold" if args.splitter == "stratified" and StratifiedGroupKFold is not None else "GroupKFold",
        "outer_folds": args.outer_folds,
        "stable_flag_filter": args.stable_flag,
        "random_seed": 20261013,
        "data_sha256": sha256(args.features.resolve()),
        "n_samples_after_filter": int(len(y)),
        "oof_complete": bool(np.all(np.isfinite(oof))),
        "target_class_counts": {str(int(level)): int(np.sum(y == level)) for level in levels},
        "folds": fold_rows,
        "pooled_oof": pooled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    original_indices = np.arange(len(data["cycle"]))
    if args.stable_flag != "all":
        original_indices = original_indices[keep]
    oof_path = args.output.with_name(args.output.stem + "_oof.csv")
    with oof_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "block_id", "stable", "y_true_bar", "y_pred_bar", "nearest_level_bar", "fold", "model"])
        nearest = nearest_level(oof, levels)
        for i, original in enumerate(original_indices):
            writer.writerow([int(data["cycle"][original]), int(data["block_id"][original]), int(data["stable"][original]), int(y[i]), float(oof[i]), float(nearest[i]), int(oof_fold[i]), oof_model[i]])
    print(json.dumps({"sensor_set": args.sensor_set, "pooled_oof": pooled}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
