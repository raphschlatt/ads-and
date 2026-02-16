# Data Contracts

## mentions.parquet

Required columns:

- `mention_id` (str)
- `bibcode` (str)
- `author_idx` (int)
- `author_raw` (str)
- `title` (str)
- `abstract` (str)
- `year` (int or null)
- `source_type` (str)
- `block_key` (str)

Optional columns:

- `orcid` (str, LSPO)
- `aff` (str)

## pairs_*.parquet

- `pair_id` (str)
- `mention_id_1` (str)
- `mention_id_2` (str)
- `block_key` (str)
- `split` (str)
- `label` (int, LSPO only)

## pair_scores.parquet

- `pair_id` (str)
- `mention_id_1` (str)
- `mention_id_2` (str)
- `block_key` (str)
- `cosine_sim` (float)
- `distance` (float)

## clusters.parquet

- `mention_id` (str)
- `block_key` (str)
- `author_uid` (str)

## publication_authors.parquet

- `bibcode` (str)
- `author_idx` (int)
- `mention_id` (str)
- `author_uid` (str)
- `source_type` (str)

## train_manifest.json (03)

- `best_checkpoint` (str)
- `best_threshold` (float)
- `best_val_f1` (float)
- `best_test_f1` (float, canonical final pairwise metric)
- `best_test_metrics` (dict: `f1`, `precision`, `recall`, `accuracy`)
- `best_val_class_counts` (dict: `pos`, `neg`)
- `best_test_class_counts` (dict: `pos`, `neg`)

## stage_metrics.json (05)

- `lspo_pairwise_f1` (float, canonical test-F1)
- `lspo_pairwise_f1_val` (float, validation F1 for diagnostics)
- `lspo_pairwise_f1_source` (`best_test_f1` or legacy fallback source)
- `threshold` (float)
- `threshold_selection_status` (str)
- `split_balance_status` (str; e.g., `ok`, `split_balance_degraded`, `split_balance_infeasible`)
- `pair_score_range_ok` (bool)
- `singleton_ratio` (float)
- `split_high_sim_rate` (float)
- `split_high_sim_rate_probe` (float)
- `merged_low_conf_rate` (float)
- `merged_low_conf_rate_probe` (float)
- `eps_boundary_hit` (bool or null)
- `eps_boundary_side` (`min`/`max` or null)
- `eps_n_valid_candidates` (int or null)
- `eps_f1_gap_best_second` (float or null)
- `counts.ads_clusters` (int; canonical unique `author_uid`)
- `counts.ads_cluster_assignments` (int; legacy count of mention->UID rows)
- `counts.ads_blocks` (int)

## cluster_qc.json (04)

- `cluster_count` (int)
- `singleton_ratio` (float)
- `n_pairs_evaluated` (int)
- `split_high_sim_count` / `split_high_sim_rate` (threshold-based)
- `merged_low_conf_count` / `merged_low_conf_rate` (threshold-based)
- `probe_threshold` (float; fixed comparability threshold, default `0.35`)
- `split_high_sim_count_probe` / `split_high_sim_rate_probe` (probe-threshold based)
- `merged_low_conf_count_probe` / `merged_low_conf_rate_probe` (probe-threshold based)

## go_no_go.json (05)

- `go` (bool)
- `checks` (list with `name`, `passed`, `detail`, `severity`)
- `blockers` (list[str]; failed blocker checks)
- `warnings` (list[str]; failed warning checks)
