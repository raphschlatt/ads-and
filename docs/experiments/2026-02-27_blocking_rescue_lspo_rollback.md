# Blocking/Rescue LSPO Experiment Log and Rollback Decision (2026-02-27)

## 1) Run Context

- Date (UTC): `2026-02-27T13:50:34Z`
- Branch: `main`
- Git HEAD: `5c7ec28`
- Model run id: `full_20260218T111506Z_cli02681429`
- Evaluation split: LSPO test (`seeds=[1,2,3,4,5]`)
- Main decision variant: `dbscan_with_constraints`

## 2) Experiment Matrix

This rollback review covers these report tags:

1. `blk_phrase_v1`
2. `blk_phrase_v1_rescue_particles_v1`
3. `blk_phrase_v1_rescue_particles_v1_relaxed80`

Baseline reference:

- `artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report.json`

## 3) Gate Policy

Promotion gate (LSPO-only):

- `delta_f1 > 0`
- `precision_drop <= 0.001` (equivalent: `delta_precision >= -0.001`)

Safety/runtime gate:

- pair count must stay stable (no material pair explosion)

## 4) Results (`dbscan_with_constraints`)

### Baseline

- `f1=0.9702503535`
- `precision=0.9635888010`
- `recall=0.9770154570`
- `accuracy=0.9439566733`
- `n_pairs=1679704`

### Candidate: `blk_phrase_v1`

- `f1=0.9701047399` (`delta=-0.0001456137`)
- `precision=0.9633005528` (`delta=-0.0002882482`)
- `recall=0.9770166538` (`delta=+0.0000011967`)
- `accuracy=0.9436986247` (`delta=-0.0002580486`)
- `n_pairs=1680403` (`delta=+699`, `+0.0416%`)

Gate outcome:

- precision rule passed
- `delta_f1` rule failed (`<= 0`)
- decision: do not promote

### Candidate: `blk_phrase_v1_rescue_particles_v1`

Core quality metrics are identical to `blk_phrase_v1`:

- `f1=0.9701047399` (`delta=-0.0001456137`)
- `precision=0.9633005528` (`delta=-0.0002882482`)
- `recall=0.9770166538` (`delta=+0.0000011967`)
- `accuracy=0.9436986247` (`delta=-0.0002580486`)
- `n_pairs=1680403` (`delta=+699`, `+0.0416%`)

Rescue diagnostics (sum over 5 seeds, rescue variant rows):

- `rescue_merged_cluster_pairs=0`
- `rescue_union_operations=0`
- `rescue_rescored_pairs=1736`

Gate outcome:

- precision rule passed
- `delta_f1` rule failed (`<= 0`)
- decision: do not promote

### Candidate: `blk_phrase_v1_rescue_particles_v1_relaxed80`

Core quality metrics are also identical to `blk_phrase_v1`:

- `f1=0.9701047399` (`delta=-0.0001456137`)
- `precision=0.9633005528` (`delta=-0.0002882482`)
- `recall=0.9770166538` (`delta=+0.0000011967`)
- `accuracy=0.9436986247` (`delta=-0.0002580486`)
- `n_pairs=1680403` (`delta=+699`, `+0.0416%`)

Rescue diagnostics:

- `rescue_merged_cluster_pairs=0`
- `rescue_union_operations=0`
- `rescue_rescored_pairs=1736`

Gate outcome:

- precision rule passed
- `delta_f1` rule failed (`<= 0`)
- decision: do not promote

## 5) Final Decision

Rollback to baseline and keep production path unchanged:

- active LSPO benchmark remains baseline `06_clustering_test_report.json`
- no blocking/rescue variant promotion
- EPS workflow remains unchanged (`98_eps_experiments_manifest.*` and `98_active_baseline.json` stay authoritative)

Rationale:

1. all blocking/rescue candidates fail promotion on `delta_f1`
2. rescue variants execute but produce zero merges under tested LSPO conditions
3. pair load is stable (`+0.0416%`) and therefore not the blocking factor; the blocker is absent quality gain

## 6) Reproduction Commands

All commands run from repo root (`/home/ubuntu/Author_Name_Disambiguation`) with same run id.
They were executed on the temporary experiment patch before rollback; on baseline-clean code these extra flags are intentionally unavailable.

### Candidate reports

```bash
PYTHONPATH=. python3 -m src.cli run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml \
  --report-tag blk_phrase_v1 \
  --lspo-reblock-mode phrase_signature_v1 \
  --lspo-split-mode fixed \
  --device auto \
  --precision-mode fp32
```

```bash
PYTHONPATH=. python3 -m src.cli run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml \
  --report-tag blk_phrase_v1_rescue_particles_v1 \
  --lspo-reblock-mode phrase_signature_v1 \
  --lspo-split-mode fixed \
  --rescue-mode particles_v1 \
  --device auto \
  --precision-mode fp32
```

```bash
PYTHONPATH=. python3 -m src.cli run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml \
  --report-tag blk_phrase_v1_rescue_particles_v1_relaxed80 \
  --lspo-reblock-mode phrase_signature_v1 \
  --lspo-split-mode fixed \
  --rescue-mode particles_v1_relaxed80 \
  --device auto \
  --precision-mode fp32
```

### Gate comparison

```bash
PYTHONPATH=. python3 scripts/ops/compare_cluster_test_reports.py \
  --baseline-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report.json \
  --candidate-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__blk_phrase_v1.json \
  --variant dbscan_with_constraints \
  --min-delta-f1 0.0 \
  --max-precision-drop 0.001
```

## 7) Provenance (SHA256)

```text
94934c81d6134d2800faea0d03bd96d18ad3efcb35810244f4f90b22a7cb4b7d  06_clustering_test_report.json
0b83837441d979751ef4dc9c6d4f667549826f308ccfc714154f4da80d38e038  06_clustering_test_report__blk_phrase_v1.json
6deca7340d30893700c5ced83f07d8f09d505342b2dd27b633674af26bac22d9  06_clustering_test_report__blk_phrase_v1_rescue_particles_v1.json
51bdea0724c027e23abf84e50de9ab028c2053f07c0ca2ac1e92f59b1b753d15  06_clustering_test_report__blk_phrase_v1_rescue_particles_v1_relaxed80.json
```

Files located in:

- `artifacts/metrics/full_20260218T111506Z_cli02681429/`

## 8) Cleanup Policy Applied

This rollback keeps:

- baseline report (`06_clustering_test_report.json`)
- EPS candidate reports (`epsbkt_*`)
- EPS freeze files (`98_eps_experiments_manifest.json`, `98_eps_experiments_manifest.md`, `98_active_baseline.json`)

This rollback removes:

- all `blk_phrase_v1*` report/summary/per-seed artifacts from the metrics folder after optional archival.

Archive created before deletion:

- `/home/ubuntu/trash/blocking_rescue_rollback_archives/full_20260218T111506Z_cli02681429_blk_phrase_reports_20260227T135153Z.tar.gz`
- archived file count: `12`
