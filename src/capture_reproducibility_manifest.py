#!/usr/bin/env python3
"""Capture a machine-readable environment and file-integrity snapshot.

This script does not change the manuscript or experimental outputs.  It records
the exact runtime, the minimal dependency set, and hashes for the source/data/
result files needed to audit the frozen hydraulic analysis.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "reproducibility_manifest_20261016.json"

KEY_FILES = [
    "README.md",
    "raw/hydraulic_uci_probe/uci_hydraulic_official.zip",
    "results/hydraulic_uci_full_audit.json",
    "results/hydraulic_uci_registry_summary.json",
    "results/hydraulic_uci_cycles.csv",
    "results/hydraulic_uci_blocks.csv",
    "results/hydraulic_uci_cycle_features.csv",
    "results/hydraulic_accumulator_primary_state4.json",
    "results/hydraulic_accumulator_primary_state4_oof.csv",
    "results/hydraulic_accumulator_ordinal_primary.json",
    "results/hydraulic_accumulator_ordinal_primary_oof.csv",
    "results/repro_20261016_estimator_comparison_fair.json",
    "results/repro_20261016_estimator_comparison_fair_oof.csv",
    "results/repro_20261016_estimator_comparison_fair_fixed_oof.csv",
    "results/final_content_audit_v4_20261014.json",
    "src/audit_hydraulic_uci.py",
    "src/build_hydraulic_registry.py",
    "src/extract_hydraulic_features.py",
    "src/evaluate_hydraulic_models.py",
    "src/evaluate_hydraulic_ordinal.py",
    "src/compare_hydraulic_estimators_fair.py",
    "src/final_content_audit_v4.py",
    "src/capture_reproducibility_manifest.py",
    "scripts/reproduce_hydraulic_audit.ps1",
    "requirements-repro-20261016.txt",
    "docs/复现运行清单-20261016.md",
    "docs/项目对话与决策记录-20261016.md",
    "docs/项目记忆、对话决策与操作日志-20261016.md",
    "docs/论文正文-v4-审计型完整稿.md",
    "docs/前四项主要问题修订审计-20261016.md",
    "docs/公平分类回归比较审计-20261016.md",
    "docs/加权指标与每折分布审计-20261016.md",
    "docs/LaTeX编译与版面验证审计-20261016.md",
    "src/audit_fair_estimator_comparison_20261016.py",
    "src/audit_hydraulic_weighted_metrics_20261016.py",
    "src/revise_narrative_20261016.py",
    "src/patch_cn_linebreak_20261016.py",
    "src/audit_narrative_revision_20261016.py",
    "results/fair_estimator_comparison_audit_20261016.json",
    "results/fair_estimator_comparison_audit_20261016_oof.csv",
    "results/weighted_metrics_audit_20261016.json",
    "results/weighted_metrics_audit_20261016_fold_distribution.csv",
    "results/weighted_metrics_audit_20261016_state4_group_metrics.csv",
    "results/weighted_metrics_audit_20261016_profile_block_metrics.csv",
    "latex_final/paper_cn.tex",
    "latex_final/paper_en.tex",
    "latex_final/paper_cn.pdf",
    "latex_final/paper_en.pdf",
    "latex_final/README.md",
    "docs/叙事改写与最终编译审计-20261016.md",
    "docs/公开代码与永久归档操作指南-20261016.md",
    "docs/最终修订与目录整理审计-20261016.md",
    "results/narrative_revision_audit_20261016.json",
]

IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "sklearn",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, module_name in IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            versions[name] = getattr(module, "__version__", None)
        except Exception as exc:  # pragma: no cover - diagnostic path
            versions[name] = f"MISSING: {exc}"
    return versions


def pip_freeze() -> list[str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        return [f"UNAVAILABLE: {exc}"]
    return [line for line in completed.stdout.splitlines() if line.strip()]


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for relative in KEY_FILES:
        path = ROOT / relative
        if path.is_file():
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        else:
            missing.append(relative)
    run_output_files: dict[str, dict[str, object]] = {}
    if output.parent.name.startswith("repro_run_") and output.parent.is_dir():
        for path in sorted(output.parent.rglob("*")):
            if path.is_file() and path != output:
                run_output_files[str(path.relative_to(output.parent))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
    frozen = pip_freeze()
    manifest = {
        "schema": "hydraulic-reproducibility-manifest/v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "system": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "package_versions": package_versions(),
        "pip_freeze": frozen,
        "pip_freeze_sha256": hashlib.sha256(("\n".join(frozen) + "\n").encode()).hexdigest(),
        "minimal_requirements": "requirements-repro-20261016.txt",
        "files": files,
        "run_output_files": run_output_files,
        "missing_key_files": missing,
        "reproduction_commands": [
            "python src/audit_hydraulic_uci.py --zip raw/hydraulic_uci_probe/uci_hydraulic_official.zip --json results/hydraulic_uci_full_audit.json --csv results/hydraulic_uci_signal_audit.csv",
            "python src/build_hydraulic_registry.py --zip raw/hydraulic_uci_probe/uci_hydraulic_official.zip",
            "python src/extract_hydraulic_features.py --zip raw/hydraulic_uci_probe/uci_hydraulic_official.zip --registry results/hydraulic_uci_cycles.csv --output results/hydraulic_uci_cycle_features.csv",
            "python src/evaluate_hydraulic_models.py --task accumulator --sensor-set all --grouping state4 --output results/hydraulic_accumulator_primary_state4.json",
            "python src/evaluate_hydraulic_ordinal.py --features results/hydraulic_uci_cycle_features.csv --sensor-set pressure_power --splitter stratified --output results/hydraulic_accumulator_ordinal_primary.json",
            "python src/compare_hydraulic_estimators_fair.py --features results/hydraulic_uci_cycle_features.csv --output results/repro_20261016_estimator_comparison_fair.json --selected-oof results/repro_20261016_estimator_comparison_fair_oof.csv --fixed-oof results/repro_20261016_estimator_comparison_fair_fixed_oof.csv",
            "python src/final_content_audit_v4.py",
        ],
        "integrity_status": "PASS" if not missing else "INCOMPLETE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "integrity_status": manifest["integrity_status"], "files": len(files), "missing": missing}, ensure_ascii=False))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
