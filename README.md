# Author Name Disambiguation

`author_name_disambiguation` is a standalone package for training and running NAND-style author disambiguation on curated source datasets.

The installed surface is intentionally small:

- `run-train-stage`
- `run-cluster-test-report`
- `export-model-bundle`
- `precompute-source-embeddings`
- `run-infer-sources`
- `run-hf-compatibility-report`
- `run-specter-benchmark`
- `run-specter-hf-lab-benchmark`

The public Python API for inference surfaces is intentionally small:

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

For the benchmark-only extras used by the HF transport lab runner:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --editable ".[bench,dev]" \
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

Precompute remote SPECTER embeddings for CPU-first inference:

```bash
export HF_TOKEN=...
author-name-disambiguation precompute-source-embeddings \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/precomputed/ads_prod_current
```

Run the strict HF compatibility gate for the current bundle:

```bash
export HF_TOKEN=...
author-name-disambiguation run-hf-compatibility-report \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/compat/ads_prod_current \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1
```

Benchmark API vs CPU vs GPU SPECTER on ADS-like inputs:

```bash
export HF_TOKEN=...
author-name-disambiguation run-specter-benchmark \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/benchmarks/ads_prod_current_specter \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1
```

This writes:

- `specter_benchmark_report.json`
- `specter_benchmark_report.md`

Run the separate HF transport lab benchmark:

```bash
export HF_TOKEN=...
author-name-disambiguation run-specter-hf-lab-benchmark \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/benchmarks/ads_prod_current_specter_hf_lab \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1 \
  --profiles all \
  --concurrency-values 4,16,64
```

This writes:

- `specter_hf_lab_report.json`
- `specter_hf_lab_report.md`

## Programmatic Inference

```python
from author_name_disambiguation import (
    InferSourcesRequest,
    PrecomputeSourceEmbeddingsRequest,
    precompute_source_embeddings,
    run_infer_sources,
)

precompute_source_embeddings(
    PrecomputeSourceEmbeddingsRequest(
        publications_path="data/raw/ads/ads_prod_current/publications.parquet",
        references_path="data/raw/ads/ads_prod_current/references.parquet",
        output_root="artifacts/precomputed/ads_prod_current",
        progress=False,
    )
)

result = run_infer_sources(
    InferSourcesRequest(
        publications_path="artifacts/precomputed/ads_prod_current/publications_precomputed.parquet",
        references_path="artifacts/precomputed/ads_prod_current/references_precomputed.parquet",
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

## CPU + HF Path

The supported CPU-first MVP is:

1. precompute `precomputed_embedding` with `hf-inference + allenai/specter`
2. run `run-infer-sources` locally on CPU, ideally with `--device cpu --cluster-backend sklearn_cpu`

This keeps the expensive text-embedding step remote while the disambiguation run stays local.
The HF path is intentionally narrow in Welle 1:

- provider: `hf-inference`
- model: `allenai/specter`
- token source: `HF_TOKEN`

Promotion of the HF path is gated by `run-hf-compatibility-report`. Until that report passes for a given bundle, treat remote HF SPECTER as experimental rather than automatically bundle-compatible.

For a broader operational comparison of `hf_api` vs local CPU vs local GPU, use `run-specter-benchmark`.
That benchmark intentionally separates:

- Track A: notebook/SPECTER parity with `max_length=512`
- Track B: current bundle parity with the bundle token cap, currently `256`

The main comparison is cap-aligned with the real inference path:

- local GPU and local CPU both truncate at the track cap
- the default local CPU backend is the exact `transformers` path
- optional `onnx_fp32` is benchmarked separately when the `cpu_onnx` extra is installed
- HF remote SPECTER also uses the same client-side tokenizer truncation
- a small raw-HF probe stays in the report only as a long-text diagnostic

For local CPU inference there is one explicit runtime switch:

- `--specter-runtime-backend transformers|onnx_fp32`

Effective default is `transformers`. `onnx_fp32` is opt-in, CPU-only, and initially experimental.
Install the optional ONNX extra if you want to use or benchmark it:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --editable ".[cpu_onnx,dev]"
```

Use `run-specter-hf-lab-benchmark` when you want the separate HF-only transport study.
That lab runner intentionally measures aggressive async/pooling profiles on:

- `micro_short_repeat` for HF-style short-text max-speed behavior
- `ads_realistic_truncated` for ADS-like capped texts

The lab report separates `prod-safe` from explicitly `lab_only` / `non_production` profiles so transport tuning does not get mixed into the package-realistic benchmark numbers.

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
