# Author Name Disambiguation

`author_name_disambiguation` is a standalone package for training and running NAND-style author disambiguation on curated source datasets.

Package design priorities:

- correctness first
- then speed
- then cost
- then minimality

The intent is a lightweight package with one clear public runtime story: `gpu | cpu | hf`.

The installed surface is intentionally small:

- `infer`
- `quality-lspo`
- `train-lspo`
- `run-train-stage`
- `run-cluster-test-report`
- `export-model-bundle`
- `precompute-source-embeddings`
- `run-infer-sources`

The public Python API is intentionally small:

- `disambiguate_sources()`
- `evaluate_lspo_quality()`
- `train_lspo_model()`
- `InferSourcesRequest`
- `InferSourcesResult`
- `PrecomputeSourceEmbeddingsRequest`
- `PrecomputeSourceEmbeddingsResult`
- `precompute_source_embeddings()`
- `run_infer_sources()`

## Install

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --editable ".[dev]" \
  --torch-backend cu126
```

In this repo, prefer `uv pip` against the existing project venv instead of introducing a separate conda env or mixing `pip` and `uv`.
Treat `python -m pip check` as a diagnostic only, not as an installation path.

Repair the shared-host GPU overlay in that same venv with the repo pins:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --reinstall \
  --no-deps \
  -r requirements-gpu-cu126.txt
```

For optional local research tooling:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --editable ".[dev,research]" \
  --torch-backend cu126
```

## GPU Env Integrity

Do not start a large infer run unless all of these succeed:

```bash
source /home/ubuntu/.venv/bin/activate
python -m pip check
python -c "import cupy, cuml"
python scripts/benchmarks/cuml_e2e_smoke.py --require-gpu-backend
```

The March 10, 2026 incident was caused by mixed installers and missing repo pins. The concrete drift was:

- `cupy-cuda12x 14.0.1` with `cuda-pathfinder 1.2.2` even though CuPy requires `>=1.3.3`
- `cuda-python 12.9.5` with `cuda-bindings 12.9.4` even though CUDA Python requires `~=12.9.5`

The compatible intersection for the shared host is now pinned in [requirements-gpu-cu126.txt](/home/ubuntu/Author_Name_Disambiguation/requirements-gpu-cu126.txt):

- `torch 2.10.0+cu126` requires `cuda-bindings==12.9.4`
- `cuml-cu12 26.2.0` accepts `cuda-python>=12.9.2,<13`
- `cupy-cuda12x 14.0.1` requires `cuda-pathfinder>=1.3.3`

That is why the repair target is `cuda-python==12.9.4`, `cuda-bindings==12.9.4`, and `cuda-pathfinder==1.3.3`.
The GPU requirements file is intentionally an overlay repair spec for the existing venv, not a one-shot exact reprovision of torch plus RAPIDS.
The repair command uses `--no-deps` on purpose so `uv` does not replace Torch's working CUDA vendor wheels with a second transitive solve from RAPIDS.
If that class of error returns, repair the venv with the `uv pip` command above. Do not manually patch single CUDA or RAPIDS packages with `pip install ...`.

## Public CLI

Normal package usage first:

Show help:

```bash
author-name-disambiguation -h
```

Disambiguate one source dataset with the packaged Fixed Model Baseline:

```bash
author-name-disambiguation infer \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-dir artifacts/exports/ads_prod_current
```

`infer` defaults to `runtime=auto`, which means:

- use `gpu` when CUDA is available
- otherwise use `cpu`
- never choose `hf` automatically

The packaged Fixed Model Baseline bundle is used automatically unless you explicitly pass `--model-bundle`.

Run an LSPO Quality Run with the Fixed Model Baseline:

```bash
author-name-disambiguation quality-lspo
```

Run an LSPO Training Run:

```bash
author-name-disambiguation train-lspo
```

Expert commands remain available when you need explicit control:

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current \
  --runtime-mode gpu
```

The package baseline uses GPU for SPECTER/pair scoring and CPU for clustering. You only need `--cluster-backend` if you intentionally want to override that default.

Run CPU-only inference explicitly:

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current_cpu \
  --dataset-id ads_prod_current \
  --runtime-mode cpu
```

Run direct HF-backed inference in one package call:

```bash
export HF_TOKEN=...
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current_hf \
  --dataset-id ads_prod_current \
  --runtime-mode hf
```

`hf` creates a dedicated paid Hugging Face endpoint on demand, uses it for remote SPECTER embeddings, and deletes it at the end of the run. It requires a token with `inference.endpoints.write`.

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

Precompute remote SPECTER embeddings when you want reusable enriched source files:

```bash
export HF_TOKEN=...
author-name-disambiguation precompute-source-embeddings \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/precomputed/ads_prod_current
```

## Programmatic Inference

```python
from author_name_disambiguation import (
    disambiguate_sources,
    evaluate_lspo_quality,
    PrecomputeSourceEmbeddingsRequest,
    precompute_source_embeddings,
)

precompute_source_embeddings(
    PrecomputeSourceEmbeddingsRequest(
        publications_path="data/raw/ads/ads_prod_current/publications.parquet",
        references_path="data/raw/ads/ads_prod_current/references.parquet",
        output_root="artifacts/precomputed/ads_prod_current",
        progress=False,
    )
)

result = disambiguate_sources(
    publications_path="data/raw/ads/ads_prod_current/publications.parquet",
    references_path="data/raw/ads/ads_prod_current/references.parquet",
    output_dir="artifacts/exports/ads_prod_current",
    runtime="auto",
)

quality = evaluate_lspo_quality(
    data_root="data",
    artifacts_root="artifacts",
    raw_lspo_parquet="data/raw/lspo/LSPO_v1.parquet",
)
```

Advanced/programmatic expert entry points remain available:

```python
from author_name_disambiguation import InferSourcesRequest, run_infer_sources

result = run_infer_sources(
    InferSourcesRequest(
        publications_path="data/raw/ads/ads_prod_current/publications.parquet",
        references_path="data/raw/ads/ads_prod_current/references.parquet",
        output_root="artifacts/exports/ads_prod_current_expert",
        dataset_id="ads_prod_current",
        model_bundle="artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1",
        runtime_mode="cpu",
        progress=False,
    )
)
```

## Public Data Contract

Input fields per source record:

- required: `Bibcode`
- required: `Author`
- optional: `Year`
- optional: `Title_en` or `Title`
- optional: `Abstract_en` or `Abstract`
- optional: `Affiliation`
- optional legacy alias: `embedding`
- optional canonical field: `precomputed_embedding`

Records without `Bibcode` or `Author` are skipped during inference. Source-mirrored outputs still preserve raw rows with empty `Author` lists and add `AuthorUID=[]` plus `AuthorDisplayName=[]`.

For the current promoted NAND bundle, text embeddings are not “any 768-dim vectors”. The active contract is:

- model family: `allenai/specter`
- dimension: `768`
- text assembly: `Title [SEP] Abstract`
- pooling: first-token / CLS from `last_hidden_state[:, 0, :]`
- tokenization: truncation with `max_length=256`

Equal dimensionality alone is not a quality-compatibility guarantee.

## Runtime Modes

The public runtime story is intentionally simple:

- `--runtime-mode gpu`
- `--runtime-mode cpu`
- `--runtime-mode hf`

`gpu` uses local transformers/SPECTER on CUDA.

`cpu` prefers local `onnx_fp32` when the optional ONNX extra is installed and the backend initializes cleanly; otherwise it falls back to the exact local `transformers` CPU path.

`hf` spins up one dedicated Hugging Face Inference Endpoint, runs remote SPECTER embeddings there, and then continues with the normal local AND tail in the same run.

The HF path is intentionally narrow:

- endpoint: dedicated Hugging Face Inference Endpoint
- model: `allenai/specter`
- hardware: `AWS / eu-west-1 / Nvidia T4 / x1`
- token source: `HF_TOKEN`

The kept HF mode is the leanest standard path we found that still worked on real ADS slices. Rejected HF approaches are summarized in [hf_endpoint_t4_decision_20260401.md](/home/ubuntu/Author_Name_Disambiguation/docs/experiments/hf_endpoint_t4_decision_20260401.md).

Install the optional ONNX extra if you want the faster CPU auto-path:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --editable ".[cpu_onnx,dev]"
```

Inference outputs under `output_root`:

- `publications_disambiguated.{parquet|jsonl}`
- optional `references_disambiguated.{parquet|jsonl}`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `summary.json`
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
