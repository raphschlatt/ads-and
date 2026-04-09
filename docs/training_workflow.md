# Training Workflow

Repo-only research workflow. Training is not part of the public `ads-and` PyPI package contract.

## Scope

This workspace surface covers:

- LSPO training
- LSPO quality runs
- train-stage orchestration
- clustering reports
- model-bundle export for research and promotion workflows

Run it from a repository checkout:

```bash
python -m author_name_disambiguation_research -h
```

## Required Inputs

- `data-root`
- `artifacts-root`
- `raw-lspo-parquet`

Optional explicit overrides:

- `--raw-lspo-h5`
- `--run-config`
- `--model-config`
- `--cluster-config`
- `--gates-config`

If no override is given, the repo defaults are used.

## End-to-End Flow

1. Train a run stage.
2. Generate the clustering test report for that run.
3. Export a model bundle for inference or evaluation.

## Commands

Train:

```bash
python -m author_name_disambiguation_research run-train-stage \
  --run-stage smoke \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/mock.parquet
```

Clustering report:

```bash
python -m author_name_disambiguation_research run-cluster-test-report \
  --model-run-id smoke_20260309T120000Z_cli12345678 \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/mock.parquet
```

Bundle export:

```bash
python -m author_name_disambiguation_research export-model-bundle \
  --model-run-id smoke_20260309T120000Z_cli12345678 \
  --artifacts-root artifacts
```

## Artifact Layout

Training writes under the explicit workspace roots:

- `artifacts/metrics/<run_id>/`
- `artifacts/checkpoints/<run_id>/`
- `artifacts/models/<run_id>/bundle_v1/`
- `data/interim/`
- `data/subsets/cache/`
- `data/subsets/manifests/`

## Notes

- `run-train-stage` no longer exports a bundle implicitly
- `export-model-bundle` is the explicit bundle creation step
- repo-local workspace files such as `configs/paths.local.yaml` are repo concerns, not package concerns
