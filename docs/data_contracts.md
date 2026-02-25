# Data Contracts

## Core Tables

`mentions.parquet` required columns:

- `mention_id`
- `bibcode`
- `author_idx`
- `author_raw`
- `title`
- `abstract`
- `year`
- `source_type`
- `block_key`

optional:

- `orcid` (LSPO)
- `aff`

`pairs_*.parquet`:

- `pair_id`
- `mention_id_1`
- `mention_id_2`
- `block_key`
- `split`
- `label` (LSPO-labeled pairs)

`pair_scores.parquet`:

- `pair_id`
- `mention_id_1`
- `mention_id_2`
- `block_key`
- `cosine_sim`
- `distance`

`clusters.parquet`:

- `mention_id`
- `block_key`
- `author_uid`

`publication_authors.parquet`:

- `bibcode`
- `author_idx`
- `mention_id`
- `author_uid`
- `source_type`

## Train Artifacts (`run-train-stage`)

`00_context.json`:

- `pipeline_scope: train` (required)
- run/config/device metadata

`03_train_manifest.json`:

- `best_checkpoint`
- `best_threshold`
- `best_val_f1`
- `best_test_f1` (canonical final train metric)
- `best_test_metrics`
- `best_val_class_counts`
- `best_test_class_counts`
- `precision_mode`

`04_clustering_config_used.json`:

- `eps_resolution` (canonical + diagnostic sweep metadata)
- `cluster_config_used`

`05_stage_metrics_<stage>.json`:

- `metric_scope: train` (required)
- `lspo_pairwise_f1` (test-based canonical)
- `lspo_pairwise_f1_val`
- split feasibility and negative coverage fields
- eps diagnostics
- train counts

`05_go_no_go_<stage>.json`:

- `go`
- `checks` (`name`, `passed`, `detail`, `severity`)
- `blockers`
- `warnings`

`06_clustering_test_report.json`:

- `pipeline_scope: train`
- `model_run_id`, `run_stage`, `generated_utc`
- `source_context_path`, `train_manifest_path`, `cluster_config_used_path`
- `lspo_source_paths`, `lspo_source_fingerprint`
- `seeds_expected`, `seeds_evaluated`
- `selected_eps`, `min_samples`, `metric`
- variant aggregates:
  - `dbscan_no_constraints`
  - `dbscan_with_constraints`
  - each with `accuracy_mean/sem`, `precision_mean/sem`, `recall_mean/sem`, `f1_mean/sem`, `n_pairs_mean`, `n_pairs_total`
- `per_seed_rows` (`seed`, `checkpoint`, `threshold`, `variant`, `accuracy`, `precision`, `recall`, `f1`, `n_pairs`)
- `delta_with_constraints_minus_no_constraints`
- `status` (`ok` on successful completion; command is fail-fast on preflight mismatches)

`06_clustering_test_summary.csv`:

- one row per variant (`dbscan_no_constraints`, `dbscan_with_constraints`)
- aggregated metrics and SEM

`06_clustering_test_per_seed.csv`:

- one row per (`seed`, `variant`) evaluation

`06_clustering_test_report.md`:

- human-readable table version of the JSON report

optional compare:

- `99_compare_train_to_baseline.json`

## Infer Artifacts (`run-infer-ads`)

`00_context.json`:

- `pipeline_scope: infer` (required)
- `dataset_id`
- model source (`model_run_id` or `model_bundle_dir`)
- resolved checkpoint/threshold/eps/precision
- CPU runtime controls and resolution:
  - `cpu_sharding_mode`
  - `cpu_workers_requested`
  - `cpu_workers_effective`
  - `cpu_limit_detected`
  - `cpu_min_pairs_per_worker`
  - `cpu_target_ram_fraction`
  - `ram_budget_bytes`
- clustering backend resolution:
  - `cluster_backend_requested`
  - `cluster_backend_effective`

`01_input_summary.json`:

- dataset paths/fingerprint
- mention and block counts
- infer subset metadata (`infer_stage`, `subset_tag`, `subset_ratio`)
- cache validation fields (`mentions_cache_*`, `subset_cache_*`)

`02_preflight_infer.json`:

- `n_mentions`, `n_blocks`, `block_p95`, `block_max`
- `pair_upper_bound`
- conservative memory estimate by component
- `memory_feasible` (gate-relevant)
- CPU sharding controls and resolved worker values:
  - `cpu_sharding_mode`
  - `cpu_workers_requested`
  - `cpu_workers_effective`
  - `cpu_limit_detected`
  - `cluster_backend_requested`
  - `cluster_backend_effective`
  - `cpu_ram_budget_bytes`

`03_pairs_qc.json`:

- ADS pair build diagnostics

`04_cluster_qc.json`:

- `cluster_count`
- `singleton_ratio`
- `n_pairs_evaluated`
- threshold-based and probe-threshold-based split/merge rates
- pair-score range stats

`05_stage_metrics_infer_ads.json`:

- `metric_scope: infer` (required)
- coverage/UID checks
- cluster quality rates
- eps diagnostics
- infer context (`infer_stage`, `subset_tag`, `subset_ratio`)
- preflight carry-over (`memory_feasible`, `pair_upper_bound`)
- source-export coverage fields
- infer counts (`ads_mentions`, unique `ads_clusters`, assignment count, `ads_blocks`)

`05_go_no_go_infer_ads.json`:

- same schema as train go/no-go

optional compare:

- `99_compare_infer_to_baseline.json`

cluster exports:

- `artifacts/clusters/<run_id>/ads_clusters_infer_ads.parquet`
- `artifacts/clusters/<run_id>/publication_authors_infer_ads.parquet`

source-mirrored exports:

- `artifacts/exports/<run_id>/publications.disambiguated.jsonl`
- `artifacts/exports/<run_id>/references.disambiguated.jsonl` (optional)

Each output record preserves original fields and appends:

- `AuthorUID` (list, same length/order as `Author`, unmapped entries are `null`)

## Model Bundle Contract (v1)

Bundle layout:

- `bundle_manifest.json`
- `checkpoint.pt`
- `model_config.yaml`
- `clustering_resolved.json`

Manifest required keys:

- `bundle_schema_version` (`v1`)
- `source_model_run_id`
- `created_utc`
- `checkpoint_hash`
- `selected_eps`
- `best_threshold`
- `precision_mode`

`run-infer-ads --model-bundle <dir>` consumes this bundle directly.

## Cache Reference Contract

`00_cache_refs.json` rows:

- `artifact_type`
- `artifact_id`
- `shared_path`
- `run_path`
- `materialization_mode` (`hardlink`/`symlink`/`copy`/`existing`)
- optional `cache_schema_version` (e.g., pair-score cache `v1`/`v2`/`v3`)
