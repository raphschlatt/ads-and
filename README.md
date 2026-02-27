# Author Name Disambiguation (NAND, CLI-First)

This repo now separates the product paths strictly:

1. `run-train-stage`: train and benchmark NAND on LSPO only.
2. `run-infer-ads`: apply a trained model to ADS data only.

`run-stage` remains as a deprecated alias to train-only behavior.

## Paper-Fair Defaults

- ORCID split: `60/20/20` (`split_assignment` in `configs/runs/*.yaml`)
- Pair protocol: exclude same publication pairs (`exclude_same_bibcode: true`)
- Training: positives + explicit negatives (`InfoNCE + negative margin`)
- Clustering: DBSCAN `eps_mode: val_sweep` in `0.20..0.50`
- Boundary audit: additional diagnostic sweep `0.55..0.70` (does not change canonical selection)
- Canonical train metric: `lspo_pairwise_f1 = best_test_f1`

## Quickstart

Show commands:

```bash
python3 -m src.cli -h
```

Train (canonical):

```bash
python3 -m src.cli run-train-stage \
  --run-stage smoke \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Finaler Clustering-Testreport auf LSPO-Testsplit (per-seed + mean/SEM):

```bash
PYTHONPATH=. python3 -m src.cli run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml \
  --device auto \
  --precision-mode fp32
```

Lean-Baseline-Integritätscheck (vor Vergleichsläufen):

```bash
PYTHONPATH=. python3 scripts/ops/check_baseline_integrity.py \
  --baseline-run-id full_20260218T111506Z_cli02681429
```

`eps`-Bucket-Qualitätsexperiment ohne Retraining (isoliert, LSPO-only Vergleich):

```bash
# 1) Kandidaten-Report auf bestehendem Baseline-Train-Run erzeugen
# Wichtig: --report-tag ist Pflicht, wenn --cluster-config-override gesetzt wird.
PYTHONPATH=. python3 -m src.cli run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml \
  --cluster-config-override configs/clustering/dbscan_paper_eps_buckets_v1.yaml \
  --report-tag epsbkt_v1 \
  --device auto \
  --precision-mode fp32

# 2) Harte Gate-Entscheidung gegen Baseline (F1 up + precision safe)
PYTHONPATH=. python3 scripts/ops/compare_cluster_test_reports.py \
  --baseline-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report.json \
  --candidate-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__epsbkt_v1.json \
  --variant dbscan_with_constraints \
  --min-delta-f1 0.0 \
  --max-precision-drop 0.001
```

Ergebnisse sichern + Soft Rollback auf Baseline:

```bash
PYTHONPATH=. python3 scripts/ops/freeze_eps_experiments.py \
  --baseline-run-id full_20260218T111506Z_cli02681429
```

Das erzeugt:

- `artifacts/metrics/full_20260218T111506Z_cli02681429/98_eps_experiments_manifest.json`
- `artifacts/metrics/full_20260218T111506Z_cli02681429/98_eps_experiments_manifest.md`
- `artifacts/metrics/full_20260218T111506Z_cli02681429/98_active_baseline.json`

Hinweis: `configs/clustering/dbscan_paper_eps_buckets_v1.yaml`, `configs/clustering/dbscan_paper_eps_buckets_v2.yaml` und `configs/clustering/dbscan_paper_eps_buckets_v3.yaml` sind aktuell experimentell und nicht promoted.

Deprecated alias (same behavior, warning emitted):

```bash
python3 -m src.cli run-stage \
  --run-stage smoke \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Export deployable model bundle:

```bash
python3 -m src.cli export-model-bundle \
  --model-run-id smoke_2026... \
  --paths-config configs/paths.local.yaml
```

Infer ADS with train run id:

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-run-id full_2026... \
  --infer-stage full \
  --paths-config configs/paths.local.yaml \
  --device auto \
  --cpu-sharding auto \
  --cpu-workers auto \
  --cpu-min-pairs-per-worker 1000000 \
  --cpu-target-ram-fraction 0.6 \
  --cluster-backend auto \
  --uid-scope dataset
```

Infer ADS with model bundle:

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-bundle artifacts/models/full_2026.../bundle_v1 \
  --infer-stage mini \
  --paths-config configs/paths.local.yaml \
  --device auto \
  --cpu-sharding auto \
  --cpu-workers auto \
  --cpu-min-pairs-per-worker 1000000 \
  --cpu-target-ram-fraction 0.6 \
  --cluster-backend auto \
  --uid-scope dataset
```

Infer ADS programmatically (same core path as CLI):

```python
from src.infer_ads_api import InferAdsRequest, run_infer_ads

result = run_infer_ads(
    InferAdsRequest(
        dataset_id="my_ads_2026",
        model_run_id="full_2026...",
        infer_stage="full",
        uid_scope="dataset",
        progress=False,
    )
)
print(result.run_id, result.go, result.publications_disambiguated_path)
```

Benchmark CPU-heavy infer stages (pair-building + clustering):

```bash
PYTHONPATH=. python3 scripts/benchmarks/bench_infer_cpu_stages.py \
  --mentions-path data/cache/_shared/subsets/ads_mentions_infer_mid_seed11_target100000_cfg692bc637_src429cfdb0e85b.parquet \
  --warmup-runs 1 \
  --measure-runs 3 \
  --cpu-min-pairs-per-worker 40000 \
  --optimized-workers auto \
  --optimized-sharding on \
  --cluster-backend sklearn_cpu
```

## CPU/GPU Runtime Controls

`run-infer-ads` supports mixed execution modes. Embeddings/pair-scoring can run on GPU (`--device auto`) while pair-building/clustering can be CPU-sharded.

Default policy is GPU-first for clustering: `--cluster-backend auto` tries `cuml_gpu` first and uses CPU only as fallback.

- `--cpu-sharding {auto,on,off}`:
  - `auto`: enable only when estimated pair load is high enough.
  - `on`: force sharding if effective workers > 1.
  - `off`: force sequential CPU path.
- `--cpu-workers {auto|N}`:
  - `auto` resolves against cgroup/affinity CPU limits and pair complexity.
  - `N` caps workers to user value, CPU limit, and block constraints.
- `--cpu-min-pairs-per-worker <int>`:
  - autoscaling threshold based on estimated pair count.
  - default `1_000_000`.
- `--cpu-target-ram-fraction <float>`:
  - target fraction of available RAM used as CPU sharding budget.
  - default `0.6`.
- `--cluster-backend {auto,sklearn_cpu,cuml_gpu}`:
  - `auto` (default/recommended): use cuML DBSCAN when available and compatible, else CPU sklearn fallback.
  - `sklearn_cpu`: force paper-reference CPU backend.
  - `cuml_gpu`: request GPU DBSCAN; runtime falls back to CPU on incompatibility/failure.
- `--uid-scope {dataset,local}`:
  - `dataset` (default): writes dataset-namespaced global IDs (`<dataset-tag>::<local_uid>`).
  - `local`: keeps legacy local IDs (`<block_key>::<cluster_label>`).
- `--uid-namespace <str>`:
  - optional override for dataset namespace when `--uid-scope dataset`.
  - defaults to normalized dataset tag from `--dataset-id`.

## Runtime Modes

- `CPU only`:
  - `--device cpu --cluster-backend sklearn_cpu`.
- `GPU embeddings/scoring + CPU clustering`:
  - `--device auto --cluster-backend sklearn_cpu`.
- `GPU embeddings/scoring + GPU clustering`:
  - `--device auto --cluster-backend auto` (or `cuml_gpu`).
  - requires optional cuML dependencies and supported CUDA runtime.

## Optional cuML GPU Clustering

cuML is optional and not required for default CPU-safe operation.

Install in a dedicated environment:

```bash
python3 -m venv .venv-cuml
source .venv-cuml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
python -m pip install "cuml-cu12>=26.2,<27" "cupy-cuda12x>=14,<15"
```

Runtime behavior:

- if `cuml`/`cupy` are missing, `cluster_backend=auto` resolves to `sklearn_cpu`.
- if `cluster_backend=cuml_gpu` is requested but unavailable, runtime warns and falls back to `sklearn_cpu`.
- if GPU DBSCAN fails at runtime, execution falls back deterministically to CPU DBSCAN with warning and reason.

Smoke-check backend behavior:

```bash
PYTHONPATH=. python3 scripts/benchmarks/cuml_e2e_smoke.py
```

Require real GPU backend resolution in a cuML-enabled env:

```bash
PYTHONPATH=. python3 scripts/benchmarks/cuml_e2e_smoke.py --require-gpu-backend
```

## Performance (Measured)

Reference benchmark run on `2026-02-25` (`100k` ADS mentions subset, estimated `292,456` pairs):

- machine: `8 vCPU` host with effective cgroup CPU limit `7`.
- benchmark method: `1` warmup + `3` measured runs, medians reported.
- benchmark report: `artifacts/benchmarks/cpu_sharding_20260225T102412Z.json`.
- benchmark setting for auto-scaling: `cpu_min_pairs_per_worker=40000` (default is `1000000`; on smaller subsets default may keep auto in sequential mode).

| Stage | Baseline (`workers=1`, `sharding=off`) | Optimized (`workers=auto`, `sharding=on`) | Speedup |
|---|---:|---:|---:|
| Pair-building | 68.024s | 32.987s | 2.062x |
| Clustering | 58.652s | 42.325s | 1.386x |
| CPU total | 127.486s | 76.342s | 1.670x |

Interpretation:

- measured CPU-stage time reduction is about `40.1%`.
- gate status on this run:
  - pair-building `>=2.0x`: passed.
  - clustering `>=2.0x`: not passed.
  - CPU total `>=2.0x`: not passed.
- end-to-end speedup depends on how dominant CPU stages are in your run.
- with CPU share in `40%-80%`, Amdahl projection gives about `1.19x-1.47x` total speedup for this measured `1.670x` CPU-stage gain.

## ADS Input Contract

Place data in:

- `data/raw/ads/<dataset-id>/publications.jsonl` or `publications.json` (required)
- `data/raw/ads/<dataset-id>/references.jsonl` or `references.json` (optional)

Expected fields per record:

- `Bibcode`
- `Author` (list or string)
- `Title_en` (or `Title`)
- `Abstract_en` (or `Abstract`)
- `Year`
- `Affiliation` (also tolerates `Affilliation`)

## Outputs

Train run (`artifacts/metrics/<run_id>/`):

- `00_context.json` (`pipeline_scope: train`)
- `03_train_manifest.json`
- `04_clustering_config_used.json` (LSPO val eps resolution)
- `05_stage_metrics_<stage>.json` (`metric_scope: train`)
- `05_go_no_go_<stage>.json`
- `06_clustering_test_report.json` (final LSPO test clustering benchmark)
- `06_clustering_test_summary.csv` (variant summary: mean/SEM)
- `06_clustering_test_per_seed.csv` (seed-wise metrics)
- `06_clustering_test_report.md` (readable benchmark report)
- optional `99_compare_train_to_baseline.json`

Infer run (`artifacts/metrics/<run_id>/`):

- `00_context.json` (`pipeline_scope: infer`)
- `01_input_summary.json`
- `02_preflight_infer.json` (memory/pair complexity estimate + feasibility)
- `03_pairs_qc.json`
- `04_cluster_qc.json`
- `04_source_export_qc.json` (source export mapping coverage)
- `05_stage_metrics_infer_ads.json` (`metric_scope: infer`)
- `05_go_no_go_infer_ads.json`
- optional `99_compare_infer_to_baseline.json`

Infer cluster exports (`artifacts/clusters/<run_id>/`):

- `ads_clusters_infer_ads.parquet`
- `publication_authors_infer_ads.parquet`

Source-mirrored infer exports (`artifacts/exports/<run_id>/`):

- `publications.disambiguated.jsonl`
- `references.disambiguated.jsonl` (if references input exists)

Mapping rule:

- `mention_id` is stable from normalized mentions.
- `author_uid` is added by clustering and is dataset-namespaced by default.
- `author_uid_local` is additionally exported in parquet outputs for traceability.
- Join key is always `mention_id`.
- In source-mirrored JSON outputs, original rows are preserved and `AuthorUID` is appended parallel to `Author`.

## Caching and Resume

- Default behavior is resume/reuse.
- Use `--force` to recompute for the same `run_id`.
- Cache tools:
  - `python3 -m src.cli cache doctor`
  - `python3 -m src.cli cache purge --target <...>` (dry-run by default, add `--yes` to apply)

## Legacy Policy

- Legacy artifacts/runs remain readable.
- New writes use the new split train/infer contracts only.
- Notebooks are legacy research tooling; CLI is the canonical product interface.
