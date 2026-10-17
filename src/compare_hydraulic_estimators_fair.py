#!/usr/bin/env python3
"""Fair paired comparison of categorical and numeric pressure estimators.

Both estimators use the same cycle table, feature columns, state4 outer folds,
and inner-fold memberships.  The fixed-family analysis pairs logistic/Ridge,
RBF-SVC/SVR, and random-forest classifiers/regressors.  A separate selected
analysis lets each task choose its own objective, which is reported as
descriptive rather than as a causal or algorithmic superiority claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from evaluate_hydraulic_models import COMPONENT_LABELS, SENSOR_SETS, candidates as classification_candidates, read_csv, splits
from evaluate_hydraulic_ordinal import candidates as regression_candidates


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
OUTER_FOLDS = 5
INNER_FOLDS = 3
LEVELS = np.array([90.0, 100.0, 115.0, 130.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state4_groups(data: dict[str, np.ndarray]) -> np.ndarray:
    keys = ["|".join(str(int(data[label][i])) for label in COMPONENT_LABELS) for i in range(len(data["cycle"]))]
    return np.unique(np.asarray(keys), return_inverse=True)[1]


def fold_signature(splits_: list[tuple[np.ndarray, np.ndarray]]) -> str:
    payload = []
    for train, test in splits_:
        payload.append({"train": train.tolist(), "test": test.tolist()})
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def common_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    nearest = LEVELS[np.argmin(np.abs(prediction[:, None] - LEVELS[None, :]), axis=1)]
    ordinal_distance = np.abs(np.searchsorted(LEVELS, nearest) - np.searchsorted(LEVELS, y_true))
    return {
        "cycles": int(len(y_true)),
        "mae_bar": float(mean_absolute_error(y_true, prediction)),
        "rmse_bar": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
        "exact_level_accuracy": float(np.mean(nearest == y_true)),
        "within_one_level_accuracy": float(np.mean(ordinal_distance <= 1)),
        "mean_ordinal_distance": float(np.mean(ordinal_distance)),
        "distance_ge_2_count": int(np.sum(ordinal_distance >= 2)),
    }


def classification_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
    }


def fixed_families() -> list[tuple[str, object, object]]:
    return [
        (
            "linear",
            Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=10.0, max_iter=2500, random_state=SEED))]),
            Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=0.1))]),
        ),
        (
            "rbf",
            Pipeline([("scale", StandardScaler()), ("model", SVC(C=10.0, kernel="rbf", gamma="scale"))]),
            Pipeline([("scale", StandardScaler()), ("model", SVR(C=10.0, kernel="rbf", gamma="scale", epsilon=0.1))]),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=SEED, n_jobs=-1),
            RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=SEED, n_jobs=-1),
        ),
    ]


def selected_pair(
    X: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    groups: np.ndarray,
    cycles: np.ndarray,
    block_ids: np.ndarray,
    output_csv: Path,
) -> dict:
    outer = splits(X, y_class, groups, OUTER_FOLDS, SEED)
    outer_sig = fold_signature(outer)
    cls_oof = np.full(len(y_class), -1.0)
    reg_oof = np.full(len(y_reg), np.nan)
    cls_model = [""] * len(y_class)
    reg_model = [""] * len(y_reg)
    oof_fold = np.full(len(y_class), -1, dtype=int)
    folds = []
    for fold, (train, test) in enumerate(outer, 1):
        cls_inner = splits(X[train], y_class[train], groups[train], INNER_FOLDS, SEED + fold)
        reg_inner = splits(X[train], y_reg[train], groups[train], INNER_FOLDS, SEED + fold)
        if fold_signature(cls_inner) != fold_signature(reg_inner):
            raise AssertionError("classification and regression inner folds differ")
        cls_scores = []
        for name, params, model in classification_candidates():
            scores = []
            for inner_train, inner_valid in cls_inner:
                fitted = clone(model).fit(X[train][inner_train], y_class[train][inner_train])
                scores.append(float(balanced_accuracy_score(y_class[train][inner_valid], fitted.predict(X[train][inner_valid]))))
            cls_scores.append({"model": name, "params": params, "score": float(np.mean(scores)), "metric": "balanced_accuracy"})
        reg_scores = []
        for name, params, model in regression_candidates():
            scores = []
            for inner_train, inner_valid in reg_inner:
                fitted = clone(model).fit(X[train][inner_train], y_reg[train][inner_train])
                scores.append(float(mean_absolute_error(y_reg[train][inner_valid], fitted.predict(X[train][inner_valid]))))
            reg_scores.append({"model": name, "params": params, "score": float(np.mean(scores)), "metric": "mae_bar"})
        best_cls = max(cls_scores, key=lambda row: row["score"])
        best_reg = min(reg_scores, key=lambda row: row["score"])
        cls_template = next(model for name, params, model in classification_candidates() if name == best_cls["model"] and params == best_cls["params"])
        reg_template = next(model for name, params, model in regression_candidates() if name == best_reg["model"] and params == best_reg["params"])
        cls_pred = clone(cls_template).fit(X[train], y_class[train]).predict(X[test]).astype(float)
        reg_pred = clone(reg_template).fit(X[train], y_reg[train]).predict(X[test])
        cls_oof[test], reg_oof[test] = cls_pred, reg_pred
        oof_fold[test] = fold
        for index in test:
            cls_model[index] = f"{best_cls['model']}:{best_cls['params']}"
            reg_model[index] = f"{best_reg['model']}:{best_reg['params']}"
        folds.append({
            "fold": fold,
            "test_cycles": int(len(test)),
            "test_groups": int(len(set(groups[test]))),
            "inner_signature": fold_signature(cls_inner),
            "classification_selected": best_cls,
            "regression_selected": best_reg,
            "classification": classification_metrics(y_class[test], cls_pred),
            "classification_mapped_to_bar": common_metrics(y_reg[test], cls_pred),
            "numeric_regression": common_metrics(y_reg[test], reg_pred),
        })
    if np.any(cls_oof < 0) or np.any(~np.isfinite(reg_oof)):
        raise AssertionError("incomplete paired OOF predictions")
    rows = []
    for i in range(len(y_class)):
        rows.append([int(cycles[i]), int(block_ids[i]), int(groups[i]), int(y_class[i]), float(cls_oof[i]), float(reg_oof[i]), int(oof_fold[i]), cls_model[i], reg_model[i]])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cycle", "block_id", "state4_id", "y_true_bar", "classification_pred_bar", "regression_pred_bar", "fold", "classification_model", "regression_model"])
        writer.writerows(rows)
    return {
        "outer_signature": outer_sig,
        "classification": {**classification_metrics(y_class, cls_oof), "mapped_to_bar": common_metrics(y_reg, cls_oof)},
        "numeric_regression": common_metrics(y_reg, reg_oof),
        "regression_out_of_range": {
            "below_90": int(np.sum(reg_oof < LEVELS[0])),
            "above_130": int(np.sum(reg_oof > LEVELS[-1])),
            "total": int(np.sum((reg_oof < LEVELS[0]) | (reg_oof > LEVELS[-1]))),
        },
        "folds": folds,
        "oof_csv": str(output_csv.relative_to(ROOT)),
        "oof_csv_sha256": sha256(output_csv),
    }


def fixed_family_pair(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cycles: np.ndarray, block_ids: np.ndarray, output_csv: Path) -> dict:
    outer = splits(X, y.astype(int), groups, OUTER_FOLDS, SEED)
    rows = []
    summaries = []
    for family, cls_template, reg_template in fixed_families():
        cls_oof = np.full(len(y), -1.0)
        reg_oof = np.full(len(y), np.nan)
        oof_fold = np.full(len(y), -1, dtype=int)
        for fold, (train, test) in enumerate(outer, 1):
            cls_oof[test] = clone(cls_template).fit(X[train], y[train].astype(int)).predict(X[test])
            reg_oof[test] = clone(reg_template).fit(X[train], y[train]).predict(X[test])
            oof_fold[test] = fold
        if np.any(cls_oof < 0) or np.any(~np.isfinite(reg_oof)):
            raise AssertionError(f"incomplete fixed-family OOF predictions: {family}")
        summaries.append({
            "family": family,
            "classification": {**classification_metrics(y.astype(int), cls_oof.astype(int)), "mapped_to_bar": common_metrics(y, cls_oof)},
            "numeric_regression": common_metrics(y, reg_oof),
            "regression_out_of_range": int(np.sum((reg_oof < LEVELS[0]) | (reg_oof > LEVELS[-1]))),
        })
        for i in range(len(y)):
            rows.append([family, int(cycles[i]), int(block_ids[i]), int(groups[i]), int(y[i]), float(cls_oof[i]), float(reg_oof[i]), int(oof_fold[i])])
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["family", "cycle", "block_id", "state4_id", "y_true_bar", "classification_pred_bar", "regression_pred_bar", "fold"])
        writer.writerows(rows)
    return {"outer_signature": fold_signature(outer), "families": summaries, "oof_csv": str(output_csv.relative_to(ROOT)), "oof_csv_sha256": sha256(output_csv)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "results/hydraulic_uci_cycle_features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/hydraulic_accumulator_estimator_comparison_fair_20261016.json")
    parser.add_argument("--selected-oof", type=Path, default=ROOT / "results/hydraulic_accumulator_estimator_comparison_fair_20261016_oof.csv")
    parser.add_argument("--fixed-oof", type=Path, default=ROOT / "results/hydraulic_accumulator_estimator_comparison_fair_20261016_fixed_oof.csv")
    args = parser.parse_args()
    fields, data = read_csv(args.features.resolve())
    feature_names = [field for field in fields if SENSOR_SETS["pressure_power"](field)]
    if len(feature_names) != 70:
        raise ValueError(f"expected 70 pressure/power features, found {len(feature_names)}")
    X = np.column_stack([data[name] for name in feature_names])
    y = data["accumulator"].astype(float)
    groups = state4_groups(data)
    if len(y) != 2205 or not np.isfinite(X).all() or set(y) != set(LEVELS):
        raise ValueError("unexpected feature table shape, finite values, or target levels")
    selected = selected_pair(X, y.astype(int), y, groups, data["cycle"].astype(int), data["block_id"].astype(int), args.selected_oof.resolve())
    fixed = fixed_family_pair(X, y, groups, data["cycle"].astype(int), data["block_id"].astype(int), args.fixed_oof.resolve())
    result = {
        "analysis_status": "paired fairness audit; both analyses use identical state4 outer folds, while only the selected analysis uses identical inner-fold memberships",
        "task": "accumulator pressure-level estimation",
        "feature_set": "pressure_power",
        "feature_count": len(feature_names),
        "n_cycles": len(y),
        "levels_bar": LEVELS.astype(int).tolist(),
        "grouping": "state4",
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "seed": SEED,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "input_sha256": sha256(args.features.resolve()),
        "selected_task_specific_objectives": {"classification": "balanced_accuracy", "regression": "MAE in bar"},
        "selected_comparison": selected,
        "fixed_matched_families": fixed,
        "interpretation_boundary": "Common metrics are paired descriptive comparisons. Classification is a discrete estimator mapped to nominal bar values; regression is a numeric estimator. No p-value or claim of universal algorithm superiority is inferred.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": selected["classification"], "regression": selected["numeric_regression"], "fixed_families": len(fixed["families"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
