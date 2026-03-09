# Author Name Disambiguation

`author_name_disambiguation` is a standalone package for training and running NAND-style author disambiguation on curated source datasets.

The installed surface is intentionally small:

- `run-train-stage`
- `run-cluster-test-report`
- `export-model-bundle`
- `run-infer-sources`

The public Python API is inference-only:

- `InferSourcesRequest`
- `InferSourcesResult`
- `run_infer_sources()`

## Install

```bash
python -m pip install -e .[dev]
```

For GPU-backed SPECTER and NAND inference, install a CUDA-enabled PyTorch build explicitly.
TensorFlow seeing a GPU is not sufficient because the SPECTER and pair-scoring paths use PyTorch.
In this repo, prefer `uv pip` against the existing project venv instead of introducing a separate conda env.

Verified repair command for the shared `/home/ubuntu/.venv` on the A100 host:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu126 \
  --reinstall "torch==2.10.0+cu126"
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

Do not start a large infer run unless that check prints `torch.cuda.is_available True`.
`run-infer-sources --device auto` now reports the fallback reason in the logs and JSON reports when PyTorch cannot use CUDA.

For optional local research tooling:

```bash
python -m pip install -r requirements-research.txt
```

## Public CLI

Show help:

```bash
author-name-disambiguation -h
```

Train a stage from explicit workspace paths:

```bash
author-name-disambiguation run-train-stage \
  --run-stage smoke \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/mock.parquet
```

Write the final clustering report for a trained run:

```bash
author-name-disambiguation run-cluster-test-report \
  --model-run-id smoke_20260309T120000Z_cli12345678 \
  --data-root data \
  --artifacts-root artifacts \
  --raw-lspo-parquet data/raw/lspo/mock.parquet
```

Export a model bundle:

```bash
author-name-disambiguation export-model-bundle \
  --model-run-id smoke_20260309T120000Z_cli12345678 \
  --artifacts-root artifacts
```

Run source inference:

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1
```

## Programmatic Inference

```python
from author_name_disambiguation import InferSourcesRequest, run_infer_sources

result = run_infer_sources(
    InferSourcesRequest(
        publications_path="data/raw/ads/ads_prod_current/publications.parquet",
        references_path="data/raw/ads/ads_prod_current/references.parquet",
        output_root="artifacts/exports/ads_prod_current",
        dataset_id="ads_prod_current",
        model_bundle="artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1",
        progress=False,
    )
)
```

## Public Data Contract

Input fields per source record:

- required: `Bibcode`
- required: `Author`
- required: `Year`
- required: `Title_en` or `Title`
- required: `Abstract_en` or `Abstract`
- optional: `Affiliation`
- optional: `embedding` or `precomputed_embedding` as a 768-dim text embedding

Inference outputs under `output_root`:

- `publications_disambiguated.{parquet|jsonl}`
- optional `references_disambiguated.{parquet|jsonl}`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

The disambiguated source files keep all input columns and add:

- `AuthorUID`
- `AuthorDisplayName`

## Docs

- [Training Workflow](docs/training_workflow.md)
- [Inference Workflow](docs/inference_workflow.md)
- [Data Contracts](docs/data_contracts.md)
- [Provenance](docs/provenance.md)
