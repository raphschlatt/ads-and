# Inference Workflow

Public inference is source-based and bundle-based.

## Required Inputs

- curated `publications`
- optional curated `references`
- exported `model_bundle`
- explicit `output_root`
- explicit `dataset_id`

## Minimal Command

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1
```

## Optional Controls

- `--infer-stage smoke|mini|mid|full`
- `--cluster-config <yaml>`
- `--gates-config <yaml>`
- `--device auto|cpu|cuda`
- `--precision-mode fp32|amp_bf16`
- `--cluster-backend auto|sklearn_cpu|cuml_gpu`
- `--uid-scope dataset|local|registry`
- `--uid-namespace <name>`

## GPU Notes

- `--device auto` may fall back to CPU if PyTorch cannot use CUDA in the current venv/session.
- The fallback is reported in `00_context.json`, `02_preflight_infer.json`, and `05_stage_metrics_infer_sources.json`.
- TensorFlow logging a visible GPU does not prove that SPECTER is on GPU; SPECTER and pair scoring use PyTorch.
- On the shared A100 host, prefer repairing the existing `/home/ubuntu/.venv` with `uv pip` instead of creating a separate conda env:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu126 \
  --reinstall "torch==2.10.0+cu126"
```

- Verify PyTorch directly before large runs:

```bash
source /home/ubuntu/.venv/bin/activate
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.cuda.device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
PY
```

- Do not trust TensorFlow-only GPU logs as proof of SPECTER acceleration.
- If `torch.cuda.is_available` is `False`, fix the venv before launching a full run.

## Output Contract

Each successful run writes:

- `publications_disambiguated.{parquet|jsonl}`
- optional `references_disambiguated.{parquet|jsonl}`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

The source-mirrored outputs preserve all input fields and add:

- `AuthorUID`
- `AuthorDisplayName`

## Notes

- The public inference path does not accept `model_run_id`.
- The package does not expect repo-relative workspace discovery at runtime.
- Packaged defaults are used when `cluster_config` and `gates_config` are omitted.
- Aborted runs should be cleaned per `output_root`; no shared cache cleanup is done automatically.
