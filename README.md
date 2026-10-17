# Hydraulic Split Audit

This repository contains the smallest reproducible package for the paper **How Split Protocols Change Reported Performance on a Public Hydraulic Test Rig: An Audit of Accumulator-Level Identification and Nominal-Pressure Error**.

The study is a bounded application evaluation on one public hydraulic test rig. It audits how cycle-level, state-grouped evaluation and estimator output spaces affect reported performance for four observed accumulator pressure levels (90, 100, 115, and 130 bar). It does not claim a new algorithm, cross-device generalisation, continuous pressure sensing, or industrial field benefit.

## Data

The raw data are the UCI Machine Learning Repository **Condition Monitoring of Hydraulic Systems** dataset: [official page](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems), DOI [10.24432/C5CW21](https://doi.org/10.24432/C5CW21), CC BY 4.0. The raw archive is intentionally not redistributed here. Download it from the official page and keep the ZIP outside this repository.

The expected archive SHA-256 is:

```text
24128AAD2EE45EEA7E6B63EBBD9992CDF25D0483A2CEBEFBFC13BC69079AF1F2
```

## Environment and reproduction

Use Python 3.14.3 or a compatible CPython release and install the pinned minimal dependencies:

```powershell
python -m pip install -r environment/requirements-repro-20261016.txt
```

Run the full data-to-model chain from the repository root. Replace the path with the downloaded official ZIP:

```powershell
& .\scripts\reproduce_hydraulic_audit.ps1 -DataZip "D:\data\uci_hydraulic_official.zip"
```

Use `-SkipModels` to rebuild only the source audit, cycle registry, and features. Outputs are written to a new `results/repro_run_<timestamp>/` directory and never overwrite the frozen result files.

## Repository layout

- `src/`: source audit, registry, feature extraction, nested classification, ordered regression, and paired estimator comparison.
- `scripts/`: one-command PowerShell reproduction entry point.
- `results/`: frozen derived tables, OOF predictions, audit summaries, and file-integrity records used by the manuscript.
- `figures/`: the two result figures used in the manuscript.
- `latex/`: Chinese and English LaTeX sources and compiled PDFs.
- `environment/`: minimal dependency specification.
- `docs/`: the reproduction checklist and a concise project decision record.

The release package excludes raw data, personal paths, browser data, user-provided documents, unrelated candidate datasets, historical drafts, and temporary files.

## Reproducibility boundary

The scripts verify the computational chain from the downloaded archive to the reported derived results. A successful run does not establish independent-device validation, unseen pressure levels, or field performance. Results are descriptive and dataset-specific.

## Citation

Please cite the paper and the dataset. Machine-readable citation metadata are provided in `CITATION.cff`. The code is released under the MIT License.

