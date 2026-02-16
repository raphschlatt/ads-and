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
