# Training Workflow

This package keeps training as an official product path, but only through the public operator CLI.

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

If no override is given, packaged defaults are used.

## End-to-End Flow

1. Train a run stage.
2. Generate the clustering test report for that run.
3. Export a model bundle for inference.

## Commands

Train:

```bash
author-name-disambiguation run-train-stage \
  --run-stage smoke \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/mock.parquet
```

Clustering report:

```bash
author-name-disambiguation run-cluster-test-report \
  --model-run-id smoke_20260309T120000Z_cli12345678 \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/mock.parquet
```

Bundle export:

```bash
author-name-disambiguation export-model-bundle \
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

- `run-train-stage` no longer exports a bundle implicitly.
- `export-model-bundle` is the only public bundle creation step.
- Repo-local path files such as `configs/paths.local.yaml` are not part of the public contract.
