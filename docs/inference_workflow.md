# Inference Workflow

Public inference is source-based and bundle-based.

## Required Inputs

- curated `publications`
- optional curated `references`
- exported `model_bundle`
- explicit `output_root`
- explicit `dataset_id`

## CPU-First Flow

For laptop users without a local GPU, the intended path is:

1. precompute text embeddings into `precomputed_embedding`
2. run normal source inference on those enriched source files

Example:

```bash
export HF_TOKEN=...
author-name-disambiguation precompute-source-embeddings \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/precomputed/ads_prod_current

author-name-disambiguation run-infer-sources \
  --publications-path artifacts/precomputed/ads_prod_current/publications_precomputed.parquet \
  --references-path artifacts/precomputed/ads_prod_current/references_precomputed.parquet \
  --output-root artifacts/exports/ads_prod_current_cpu \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1 \
  --runtime-mode cpu \
  --cluster-backend sklearn_cpu
```

The remote HF path is intentionally fixed to:

- model: `allenai/specter`
- endpoint: dedicated Hugging Face Inference Endpoint
- hardware: `AWS / eu-west-1 / Nvidia T4 / x1`
- env var: `HF_TOKEN` with `inference.endpoints.write`

## Embedding Contract

The current productive NAND bundle expects this text embedding contract:

- `precomputed_embedding` is the canonical field name
- `embedding` remains a legacy alias on input
- model family: `allenai/specter`
- dimension: `768`
- text assembly: `Title [SEP] Abstract`
- pooling: first-token / CLS from `last_hidden_state[:, 0, :]`
- tokenization: `truncation=True`, `max_length=256`

Do not treat “same dimension” as compatibility. A different embedding family can fit into the tensor shape and still be semantically wrong for the active bundle.

## Minimal Command

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1
```

## Runtime Modes

The public runtime interface is:

- `--runtime-mode gpu`
- `--runtime-mode cpu`
- `--runtime-mode hf`

`gpu` uses local transformers/SPECTER on CUDA.

`cpu` prefers local `onnx_fp32` when available and falls back to the exact local `transformers` CPU path if ONNX is unavailable or fails to initialize.

`hf` creates one dedicated Hugging Face endpoint, uses it for remote SPECTER embeddings, and then continues with the normal local AND tail in the same run.

This mode is paid, requires endpoint-write permission on the token, and is intentionally fixed to one standard spec so the package stays small.

Rejected HF alternatives are recorded once in [hf_endpoint_t4_decision_20260401.md](/home/ubuntu/Author_Name_Disambiguation/docs/experiments/hf_endpoint_t4_decision_20260401.md).

## Optional Controls

- `--infer-stage smoke|mini|mid|full`
- `--cluster-config <yaml>`
- `--gates-config <yaml>`
- `--precision-mode fp32|amp_bf16`
- `--cluster-backend auto|sklearn_cpu|cuml_gpu`
- `--uid-scope dataset|local|registry`
- `--uid-namespace <name>`

## GPU Notes

- `--device auto` may fall back to CPU if PyTorch cannot use CUDA in the current venv/session.
- The fallback is reported in `00_context.json`, `02_preflight_infer.json`, and `05_stage_metrics_infer_sources.json`.
- TensorFlow logging a visible GPU does not prove that SPECTER is on GPU; SPECTER and pair scoring use PyTorch.
- The canonical GPU environment for this repo is `/home/ubuntu/Author_Name_Disambiguation/.venv`.
- Bootstrap that repo venv with `uv pip` instead of creating a second conda env or patching CUDA packages ad hoc:

```bash
source /home/ubuntu/Author_Name_Disambiguation/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/Author_Name_Disambiguation/.venv/bin/python \
  --editable ".[dev]" \
  --torch-backend cu126
```

- Repair the repo GPU overlay in that same venv from the repo pins:

```bash
source /home/ubuntu/Author_Name_Disambiguation/.venv/bin/activate
uv pip install \
  --python /home/ubuntu/Author_Name_Disambiguation/.venv/bin/python \
  --reinstall \
  --no-deps \
  -r requirements-gpu-cu126.txt
```

- Verify the full GPU environment directly before large runs:

```bash
source /home/ubuntu/Author_Name_Disambiguation/.venv/bin/activate
python -m pip check
python scripts/ops/gpu_env_doctor.py --json
```

- Do not trust TensorFlow-only GPU logs as proof of SPECTER acceleration.
- If `gpu_env_doctor.py` reports `tensorflow_expected_cu12_but_detected_cu13_stack`, `chars2vec` will run on CPU even while PyTorch still uses CUDA.
- `requirements-gpu-cu126.txt` is the repo-managed `cu126/cu12` overlay for:
  - `torch 2.10.x + cu126`
  - TensorFlow `2.20` GPU vendor wheels on `cu12`
- The repair command uses `--no-deps` on purpose so `uv` does not replace the working Torch CUDA wheels during runtime repair.
- `python -m pip check` is a diagnostic, not an install path.
- Do not manually patch single CUDA or TensorFlow packages with `pip install ...`; always repair from `requirements-gpu-cu126.txt` with `uv pip`.
- `cuml_e2e_smoke.py` is an optional smoke test for a separate RAPIDS/cuML environment and is no longer part of the standard repo-`.venv` gate for `infer_sources`.

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

## Baseline, Quality Reference, And Cleanup

There are now two deliberately separate references:

- Active ADS infer baseline:
  - `docs/baselines/infer_ads_active.json`
  - `docs/baselines/infer_ads_full_run_20260305_v22_fix2.json`
  - full artifact keep-set: `artifacts/exports/bench_full_v22_fix2`
- Operational LSPO quality reference:
  - `docs/baselines/lspo_quality_operational.json`
  - current reproducible compat state: `srcb2c9203fe342`
  - current operational report: `06_clustering_test_report__chars_cpu_20260407_v1.json`

Important distinction:

- `srcd52...` is historical/advisory only
- `srcb2...` is the currently available and reproducible LSPO comparison state
- the latest accepted cold-run package optimization session is documented in `docs/experiments/infer_cold_path_wave_20260408.md`
- that experiment record is not the same thing as a formal ADS baseline promotion

Default integrity checks now validate the operational LSPO reference:

```bash
source /home/ubuntu/Author_Name_Disambiguation/.venv/bin/activate
python scripts/ops/check_baseline_integrity.py
```

Use the historical check only when you explicitly want to audit the old train baseline assumptions:

```bash
source /home/ubuntu/Author_Name_Disambiguation/.venv/bin/activate
python scripts/ops/check_baseline_integrity.py --mode historical
```

### Current Retention Policy

This cleanup policy is intentionally **repo-only**. It does not delete `data/raw`, so raw training/source inputs stay intact.

Keep full:

- `artifacts/exports/bench_full_v22_fix2`

Keep in `product-only` form:

- `artifacts/exports/ads_full_chars2vec_cpu_ab_20260407`

Keep in `json-only` form:

- `artifacts/exports/ads_full_speedup_wave_20260407_v1`
- `artifacts/exports/ads_full_chars2vec_gpu_repaired_20260407`
- `artifacts/exports/ads_full_package_auto_20260407`
- `artifacts/exports/full_gpu_canonical_fastchars_cpucluster_20260402`

Already-small runs such as `bench_full_perf_pkg1`, `bench_full_perf_pkg1_fix1`, and `bench_full_wave_b_v1` are left untouched.

### Current Size Snapshot

Measured on `2026-04-07`:

- before cleanup:
  - repo total: `51G`
  - `artifacts`: `35G`
  - `artifacts/exports`: `35G`
  - `data`: `4.9G`
  - `data/cache/_shared`: `2.6G`
  - `data/raw`: `1.2G`
  - `.venv`: `11G`
- after applying the current retention policy:
  - repo total: `24G`
  - `artifacts`: `7.8G`
  - `artifacts/exports`: `7.6G`
  - `data`: `4.8G`
  - `data/cache/_shared`: `2.5G`
  - `data/raw`: `1.2G`
  - `.venv`: `11G`

The main reclaimed space came from pruning large ADS candidate run directories while keeping one full baseline and one compact current CPU reference.

### Compare, Freeze, And Prune

Compare new ADS candidates against the active ADS baseline:

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

After the compare report exists, freeze the candidate into a formal decision record:

```bash
python scripts/ops/freeze_infer_baseline.py \
  --baseline-run-id bench_full_v22_fix2 \
  --candidate-run-id <new_run_dir_name> \
  --metrics-root artifacts/exports
```

This writes inside the candidate run dir:

- `98_infer_baseline_decision.json`
- `98_infer_baseline_decision.md`

Use `--runtime-metric-max-delta <metric>=<max_delta_seconds>` only when you want additional speed gates for a specific wave.

After a candidate is resolved, prune it explicitly:

```bash
python scripts/ops/prune_infer_run.py \
  --run-dir artifacts/exports/<failed_run_dir> \
  --mode json-only
```

For the current operational CPU reference, keep final products plus small metadata:

```bash
python scripts/ops/prune_infer_run.py \
  --run-dir artifacts/exports/ads_full_chars2vec_cpu_ab_20260407 \
  --mode product-only
```

`json-only` preserves small top-level metadata such as:

- `00_context.json`
- `01_input_summary.json`
- `02_preflight_infer.json`
- `03_pairs_qc.json`
- `04_*`
- `05_*`
- `98_*`
- `99_compare_infer_*.json`
- `summary.json`
- `*_run_consistency.json`

`product-only` keeps the same metadata plus the final top-level product parquets:

- `publications_disambiguated.parquet`
- `references_disambiguated.parquet`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`

The prune tool refuses unresolved candidates; a run must already have:

- `99_compare_infer_to_baseline.json`
- `98_infer_baseline_decision.json`
- `98_infer_baseline_decision.md`

If a candidate is promoted, write a new versioned ADS baseline manifest:

```bash
python scripts/ops/freeze_infer_baseline.py \
  --baseline-run-id bench_full_v22_fix2 \
  --candidate-run-id <promoted_run_dir> \
  --metrics-root artifacts/exports \
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

### Shared Cache Policy

The current shared keep-set is the real operational `srcb2...` compat set:

- `data/cache/_shared/subsets/lspo_mentions_full_seed11_targetfull_cfg0dbcdaf9_srcb2c9203fe342.parquet`
- `data/cache/_shared/subsets/subset_full_seed11_targetfull_cfg0dbcdaf9_srcb2c9203fe342.meta.json`
- `data/cache/_shared/embeddings/lspo_chars2vec_cpu_4b7bfbd51bb9.npy`
- `data/cache/_shared/embeddings/lspo_specter_4b7bfbd51bb9.npy`
- `data/cache/_shared/pairs/lspo_mentions_split_266b4f62be53.parquet`
- `data/cache/_shared/pairs/lspo_pairs_266b4f62be53.parquet`
- `data/cache/_shared/pairs/split_balance_266b4f62be53.json`
- `data/cache/_shared/pairs/pairs_qc_train_266b4f62be53.json`

The duplicate generic `lspo_chars2vec_4b7bfbd51bb9.npy` is not part of the keep-set and has been removed.

## Notes

- The public inference path does not accept `model_run_id`.
- The package does not expect repo-relative workspace discovery at runtime.
- Packaged defaults are used when `cluster_config` and `gates_config` are omitted.
- Aborted runs should be cleaned per `output_root`; no shared cache cleanup is done automatically.
- Large historical inference artifacts outside the documented keep-set are expected to be removed after baseline freeze.
- A candidate run is not considered resolved until it has both a compare report and a `98_infer_baseline_decision.{json,md}` record.
- `docs/baselines/infer_ads_active.json` is the machine-readable pointer to the currently active ADS baseline.
- `docs/baselines/lspo_quality_operational.json` is the machine-readable pointer for the current reproducible LSPO quality reference.
