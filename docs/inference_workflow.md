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
- On the shared A100 host, bootstrap the existing `/home/ubuntu/.venv` with `uv pip` instead of creating a separate conda env or mixing `pip` and `uv` installs:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --editable ".[dev]" \
  --torch-backend cu126
```

- Repair the RAPIDS/CUDA overlay in that same venv from the repo pins:

```bash
source /home/ubuntu/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/.venv/bin/python \
  --reinstall \
  --no-deps \
  -r requirements-gpu-cu126.txt
```

- Verify the full GPU environment directly before large runs:

```bash
source /home/ubuntu/.venv/bin/activate
python -m pip check
python -c "import cupy, cuml"
python scripts/benchmarks/cuml_e2e_smoke.py --require-gpu-backend
```

- Do not trust TensorFlow-only GPU logs as proof of SPECTER acceleration.
- If any of those checks fail, fix the venv before launching a full run.
- Root cause of the March 10, 2026 RAPIDS outage: mixed installers plus missing repo pins.
  - `cupy-cuda12x 14.0.1` was installed alongside `cuda-pathfinder 1.2.2`, although CuPy requires `>=1.3.3`.
  - `cuda-python 12.9.5` was installed alongside `cuda-bindings 12.9.4`, although CUDA Python requires `~=12.9.5`.
- Compatible repair target for the shared host:
  - `torch 2.10.0+cu126` requires `cuda-bindings==12.9.4`.
  - `cuml-cu12 26.2.0` accepts `cuda-python>=12.9.2,<13`.
  - `cupy-cuda12x 14.0.1` requires `cuda-pathfinder>=1.3.3`.
  - Therefore the repo-pinned repair set is `cuda-python==12.9.4`, `cuda-bindings==12.9.4`, `cuda-pathfinder==1.3.3`.
- `requirements-gpu-cu126.txt` is an overlay repair spec for the existing host venv, not a from-scratch exact solve for torch plus RAPIDS together.
- The repair command uses `--no-deps` on purpose so `uv` does not replace Torch's working CUDA vendor wheels with a second transitive solve from RAPIDS.
- `python -m pip check` is a diagnostic, not an install path.
- Do not manually patch single CUDA or RAPIDS packages with `pip install ...`; always repair from `requirements-gpu-cu126.txt` with `uv pip`.

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

## Baseline And Compare

The canonical inference baseline is tracked in:

- `docs/baselines/infer_ads_full_run_20260305_v21_fix4.json`

Current keep-set for large inference artifacts:

- `artifacts/exports/bench_full_v21_fix4`
- `artifacts/exports/bench_mid_v21_fix4`
- `artifacts/exports/infer_ads_full_20260305_full_20260310T134713Z`
- `artifacts/exports/infer_ads_full_20260305_full_20260310T134713Z.run.log`

The historical full reference remains for provenance. New comparisons should use the current baseline:

```bash
author-name-disambiguation compare-infer-baseline \
  --baseline-run-id bench_full_v21_fix4 \
  --current-run-id <new_run_dir_name> \
  --metrics-root artifacts/exports
```

This writes:

- `99_compare_infer_to_baseline.json`

The compare report includes:

- stage/count deltas from `05_stage_metrics_infer_sources.json`
- coverage/go-no-go deltas
- partition-aware `mention_clusters` drift, not just raw UID-string diffs

## Notes

- The public inference path does not accept `model_run_id`.
- The package does not expect repo-relative workspace discovery at runtime.
- Packaged defaults are used when `cluster_config` and `gates_config` are omitted.
- Aborted runs should be cleaned per `output_root`; no shared cache cleanup is done automatically.
- Large historical inference artifacts outside the documented keep-set are expected to be removed after baseline freeze.
