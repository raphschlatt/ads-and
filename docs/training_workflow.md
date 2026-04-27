# LSPO and Training Workflow

This is a repo-level runbook for LSPO evaluation and model-training work. These
commands are not part of the public `ads-and` package contract.

## Before You Start

- Raw LSPO is external to this repository. Download it yourself from the
  original release and point the workflow at your local copy.
- Both `--raw-lspo-parquet` and `--raw-lspo-h5` are supported. HDF5 is a
  regular input path for the Zenodo LSPO release.
- `quality-lspo` is for real local research runs with local train metrics and
  checkpoints. The packaged fixed bundle used by public `ads-and infer` is
  infer-only and does not satisfy `quality-lspo` by itself.
- The published baseline run `full_20260218T111506Z_cli02681429` has a small
  tracked repository-level reproduction pack under `artifacts/metrics/`,
  `artifacts/checkpoints/`, and `artifacts/models/`. Raw LSPO and large caches
  are not redistributed.
- Use `python -m author_name_disambiguation_research doctor --model-run-id <run_id>`
  to check whether the local LSPO source and mandatory train artifacts are in
  place before a quality run.

## Reproduce Published LSPO Baseline

The tracked baseline report behind the README LSPO row is:

```text
artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__chars_cpu_20260407_v1.json
```

It evaluates seeds 1 through 5 and reports, for DBSCAN with constraints,
F1 97.02%, precision 96.36%, and recall 97.70%. The five seed checkpoints are
tracked under:

```text
artifacts/checkpoints/full_20260218T111506Z_cli02681429/
```

The single fixed inference bundle is tracked separately under:

```text
artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1/
```

That bundle is the public inference model, not the full five-seed quality
protocol by itself.

After downloading LSPO from Zenodo into `data/raw/lspo/`, inspect the local
state:

```bash
uv run python -m author_name_disambiguation_research doctor \
  --model-run-id full_20260218T111506Z_cli02681429
```

For a full release-quality rerun, use a GPU machine. A CPU rerun is possible
but unnecessarily slow for normal release work:

```bash
uv run python -m author_name_disambiguation_research quality-lspo \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --raw-lspo-parquet data/raw/lspo/LSPO_v1.parquet \
  --report-tag release_0_1_3_py312_torch260_transformers4562
```

Compare the release candidate against the tracked baseline:

```bash
uv run python scripts/ops/compare_cluster_test_reports.py \
  --baseline-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__chars_cpu_20260407_v1.json \
  --candidate-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__release_0_1_3_py312_torch260_transformers4562.json \
  --min-delta-f1 -0.001 \
  --max-precision-drop 0.001
```

If the gate passes, keep the generated `06_clustering_test_*__release_...`
files and the matching `99_compare_cluster_report_to_baseline__release_...`
JSON as release evidence. Do not commit Raw LSPO, `data/interim`, or
`data/cache/_shared`.

## Inference-Only Experiment

For an inference-only experiment, keep the Trained NAND Model fixed and run an
LSPO Gate Run first. Do not run `run-train-stage`.

Canonical LSPO Gate Run:

```bash
python -m author_name_disambiguation_research run-cluster-test-report \
  --model-run-id <local-train-run-id> \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/LSPO_v1.parquet \
  --report-tag <experiment-tag>
```

If you only have the Zenodo HDF5 source locally, use `--raw-lspo-h5` instead:

```bash
python -m author_name_disambiguation_research run-cluster-test-report \
  --model-run-id <local-train-run-id> \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-h5 data/raw/lspo/LSPO_v1.h5 \
  --report-tag <experiment-tag>
```

This command rebuilds or verifies the LSPO evaluation subset from the Raw LSPO
Source, reuses the checkpoints and train metadata for the target local run,
recomputes embeddings, pair scores, and clustering, then writes
`06_clustering_test_report*.json`, `*.csv`, and `*.md` outputs under
`artifacts/metrics/<model-run-id>/`. It requires a real local train run under
`artifacts/metrics/<model-run-id>/` plus the referenced checkpoints; the
packaged infer bundle alone is not enough.

Convenience wrapper:

```bash
python -m author_name_disambiguation_research quality-lspo \
  --model-run-id <local-train-run-id> \
  --report-tag <experiment-tag>
```

If you only have the Zenodo HDF5 source locally:

```bash
python -m author_name_disambiguation_research quality-lspo \
  --model-run-id <local-train-run-id> \
  --raw-lspo-h5 data/raw/lspo/LSPO_v1.h5 \
  --report-tag <experiment-tag>
```

## Model-Training Experiment

Use `run-train-stage` only when the experiment intentionally changes weights,
training configuration, model architecture, or training-time representation
semantics.

```bash
python -m author_name_disambiguation_research run-train-stage \
  --run-stage full \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/LSPO_v1.parquet
```

If you are working from the Zenodo HDF5 release instead, use:

```bash
python -m author_name_disambiguation_research run-train-stage \
  --run-stage full \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-h5 data/raw/lspo/LSPO_v1.h5
```

`run-train-stage` prepares LSPO mentions, subsets, splits, pairs, embeddings,
checkpoints, threshold metadata, stage metrics, and go/no-go reports. It does
not export a model bundle implicitly.

Artifact map for `quality-lspo`:

- Mandatory local train artifacts:
  - `artifacts/metrics/<run_id>/00_context.json`
  - `artifacts/metrics/<run_id>/03_train_manifest.json`
  - `artifacts/metrics/<run_id>/04_clustering_config_used.json`
  - `artifacts/metrics/<run_id>/05_stage_metrics_<run_stage>.json`
  - every checkpoint referenced by `03_train_manifest.json`
- Reconstructable from raw LSPO plus train metadata:
  - `data/interim/lspo_mentions.parquet`
  - subset / split / pair artifacts
  - shared embeddings
  - `06_clustering_test_report.{json,csv,md}`
- Cache-like outputs:
  - shared subset manifests
  - shared pair-score / embedding caches under the workspace cache tree

Convenience wrapper:

```bash
python -m author_name_disambiguation_research train-lspo \
  --run-stage full
```

## Bundle Export

After a model-training experiment, export a model bundle explicitly:

```bash
python -m author_name_disambiguation_research export-model-bundle \
  --model-run-id <trained-run-id> \
  --artifacts-root artifacts
```

The default output is `artifacts/models/<trained-run-id>/bundle_v1/`. The
bundle contains `checkpoint.pt`, `model_config.yaml`,
`clustering_resolved.json`, and `bundle_manifest.json`.

## Workflow Rule

- Inference-only experiment: fixed Trained NAND Model -> LSPO Gate Run -> ADS Full Candidate Run.
- Model-training experiment: `run-train-stage` -> LSPO Gate Run for the new run -> optional bundle export -> optional ADS Full Candidate Run.
