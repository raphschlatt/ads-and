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
