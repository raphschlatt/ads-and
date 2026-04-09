# Inference Workflow

Repo-only operational and research reference. This document is not part of the public PyPI package contract.

## Public Product Boundary

The public package is `ads-and`:

- install with `uv pip install ads-and`
- use the bundled ADS baseline model automatically
- run local inference via `ads-and infer`
- rely on local CPU/GPU auto-selection only

This document covers the broader repo-only workspace around that product:

- explicit `run-infer-sources`
- baseline comparison and retention
- internal runtime telemetry
- explicit research/training/baseline workflows

Use the research CLI from a repo checkout:

```bash
python -m author_name_disambiguation_research run-infer-sources -h
```

## Required Inputs

- curated `publications`
- optional curated `references`
- explicit `output_root`
- explicit `dataset_id`

The public package path uses the embedded fixed ADS model automatically.
Repo-only workflows may still point at explicit bundle paths when needed.

## Minimal Repo Command

```bash
python -m author_name_disambiguation_research run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current
```

## Product Runtime Policy

The current supported product/runtime stance is:

- `chars2vec` is CPU-only in product inference
- `cluster_backend=auto` resolves to `sklearn_cpu`
- `cuml_gpu` is explicit opt-in only
- `numba` remains optional and is not auto-selected

These are deliberate product decisions, not accidental leftovers.

## Hardware-Adaptive Defaults

`run-infer-sources` resolves an internal runtime policy before expensive work starts. That policy is recorded in:

- `00_context.json`
- `02_preflight_infer.json`
- `05_stage_metrics_infer_sources.json`

Recorded blocks:

- `host_profile`
- `resolved_runtime_policy`
- `safety_fallbacks`

Current automatic behavior:

- `chars2vec`
  - always CPU in the product path
  - default batch `128`
  - reduced to `64` below `12 GiB` available RAM
  - reduced to `32` below `6 GiB` available RAM
- `SPECTER`
  - on CUDA: use the accepted GPU path and existing auto-batch heuristic
  - on CPU: use `cpu_auto`, which prefers ONNX and falls back to transformers
  - on CUDA OOM, halve batch size repeatedly down to `16` before CPU fallback
- pair scoring
  - on CUDA OOM, halve batch size repeatedly down to `1024` before CPU fallback
  - on CPU, clamp automatic score batch if the peak estimate exceeds `10%` of available RAM
- clustering
  - `auto` resolves to `sklearn_cpu`
  - `cuml_gpu` remains explicit opt-in only
- Exact-Graph union implementation
  - default is `python`
  - `numba` is not auto-selected even if installed

## CPU-Only Notes

The 2026-04-08 repo-host CPU-only smoke and ONNX A/B are retained as operational reference:

- CPU-only is supported and robust
- ONNX CPU was functional and quality-identical on that host
- ONNX CPU was not faster than the plain transformers CPU path there

Current product decision:

- keep `cpu_auto`
- do not add a host-specific ONNX-vs-transformers selector unless CPU-only becomes a first-class performance target later

## Fail-Fast Boundaries

The package falls back automatically when a supported safer path exists. It fails early only when a run is physically or contractually impossible.

Early hard failures are expected for:

- physically insufficient scratch space for exact out-of-core inference
- required artifacts or runtime components with no supported fallback
- explicit backend requests that cannot be honored and have no supported fallback

Automatic fallback is expected for:

- `device=auto` when CUDA is unavailable
- TensorFlow GPU unavailability
- ONNX CPU unavailability
- `cuml_gpu` unavailability under `cluster_backend=auto`

## Repo-Only Operations

These remain in the repo workspace and are not part of the public package contract:

- baseline compare/freeze flows
- cleanup/retention workflows
- experimental GPU clustering environments

## Related Repo Docs

- `docs/training_workflow.md`
- `docs/provenance.md`
- `docs/experiments/infer_cold_path_wave_20260408.md`
