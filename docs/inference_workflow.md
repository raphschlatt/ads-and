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

The current promoted ADS inference baseline is tracked in:

- `docs/baselines/infer_ads_full_run_20260305_v22_fix2.json`
- `docs/baselines/infer_ads_active.json`

Current keep-set for large inference artifacts:

- `artifacts/exports/bench_full_v22_fix2`

New comparisons should use the current baseline:

```bash
author-name-disambiguation compare-infer-baseline \
  --baseline-run-id bench_full_v22_fix2 \
  --current-run-id <new_run_dir_name> \
  --metrics-root artifacts/exports
```

This writes:

- `99_compare_infer_to_baseline.json`

The compare report includes:

- stage/count deltas from `05_stage_metrics_infer_sources.json`
- coverage/go-no-go deltas
- partition-aware `mention_clusters` drift, not just raw UID-string diffs

After the compare report exists, freeze the candidate into a formal promote-or-keep-baseline decision:

```bash
python scripts/ops/freeze_infer_baseline.py \
  --baseline-run-id bench_full_v22_fix2 \
  --candidate-run-id <new_run_dir_name> \
  --metrics-root artifacts/exports \
  --runtime-metric-max-delta clustering.dbscan_seconds_total=0
```

This writes inside the candidate run dir:

- `98_infer_baseline_decision.json`
- `98_infer_baseline_decision.md`

Use additional `--runtime-metric-max-delta <metric>=<max_delta_seconds>` gates to reflect the focus of the wave under test.

If a candidate is not promoted, keep the compare JSONs and prune the heavy parquet outputs:

```bash
python scripts/ops/prune_infer_run.py --run-dir artifacts/exports/<failed_run_dir>
```

The JSON-only retention set now preserves:

- `00_context.json`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`
- `98_infer_baseline_decision.json`
- `98_infer_baseline_decision.md`
- `99_compare_infer_to_baseline.json`

If a candidate is promoted, write a new versioned baseline manifest:

```bash
python scripts/ops/freeze_infer_baseline.py \
  --baseline-run-id bench_full_v22_fix2 \
  --candidate-run-id <promoted_run_dir> \
  --metrics-root artifacts/exports \
  --runtime-metric-max-delta clustering.dbscan_seconds_total=0 \
  --promote-manifest-path docs/baselines/infer_ads_full_run_20260305_<tag>.json \
  --active-baseline-path docs/baselines/infer_ads_active.json
```

If you already have a `98_infer_baseline_decision.json` and only need the versioned manifest, the lower-level helper remains available:

```bash
python scripts/ops/write_infer_baseline_manifest.py \
  --run-dir artifacts/exports/<promoted_run_dir> \
  --manifest-path docs/baselines/infer_ads_full_run_20260305_<tag>.json \
  --compare-report artifacts/exports/<promoted_run_dir>/99_compare_infer_to_baseline.json
```

## Notes

- The public inference path does not accept `model_run_id`.
- The package does not expect repo-relative workspace discovery at runtime.
- Packaged defaults are used when `cluster_config` and `gates_config` are omitted.
- Aborted runs should be cleaned per `output_root`; no shared cache cleanup is done automatically.
- Large historical inference artifacts outside the documented keep-set are expected to be removed after baseline freeze.
- A candidate run is not considered resolved until it has both a compare report and a `98_infer_baseline_decision.{json,md}` record.
- `docs/baselines/infer_ads_active.json` is the machine-readable pointer to the currently active ADS baseline.
