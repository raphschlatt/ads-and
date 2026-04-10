# LSPO and Training Workflow

This is a repo-level runbook for LSPO evaluation and model-training work. These
commands are not part of the public `ads-and` package contract.

## Inference-Only Experiment

For an inference-only experiment, keep the Trained NAND Model fixed and run an
LSPO Gate Run first. Do not run `run-train-stage`.

Canonical LSPO Gate Run:

```bash
python -m author_name_disambiguation_research run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/LSPO_v1.parquet \
  --report-tag <experiment-tag>
```

This command rebuilds or verifies the LSPO evaluation subset from the Raw LSPO
Source, reuses the existing trained model baseline, recomputes embeddings, pair
scores, and clustering, then writes `06_clustering_test_report*.json`,
`*.csv`, and `*.md` outputs under `artifacts/metrics/<model-run-id>/`.

Convenience wrapper:

```bash
python -m author_name_disambiguation_research quality-lspo \
  --model-run-id full_20260218T111506Z_cli02681429 \
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

`run-train-stage` prepares LSPO mentions, subsets, splits, pairs, embeddings,
checkpoints, threshold metadata, stage metrics, and go/no-go reports. It does
not export a model bundle implicitly.

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
