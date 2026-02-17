# Author Name Disambiguation (NAND, CLI-First)

This repo now separates the product paths strictly:

1. `run-train-stage`: train and benchmark NAND on LSPO only.
2. `run-infer-ads`: apply a trained model to ADS data only.

`run-stage` remains as a deprecated alias to train-only behavior.

## Paper-Fair Defaults

- ORCID split: `60/20/20` (`split_assignment` in `configs/runs/*.yaml`)
- Pair protocol: exclude same publication pairs (`exclude_same_bibcode: true`)
- Training: positives + explicit negatives (`InfoNCE + negative margin`)
- Clustering: DBSCAN `eps_mode: val_sweep` in `0.20..0.50`
- Boundary audit: additional diagnostic sweep `0.55..0.70` (does not change canonical selection)
- Canonical train metric: `lspo_pairwise_f1 = best_test_f1`

## Quickstart

Show commands:

```bash
python3 -m src.cli -h
```

Train (canonical):

```bash
python3 -m src.cli run-train-stage \
  --run-stage smoke \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Deprecated alias (same behavior, warning emitted):

```bash
python3 -m src.cli run-stage \
  --run-stage smoke \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Export deployable model bundle:

```bash
python3 -m src.cli export-model-bundle \
  --model-run-id smoke_2026... \
  --paths-config configs/paths.local.yaml
```

Infer ADS with train run id:

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-run-id full_2026... \
  --infer-stage full \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Infer ADS with model bundle:

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-bundle artifacts/models/full_2026.../bundle_v1 \
  --infer-stage mini \
  --paths-config configs/paths.local.yaml \
  --device auto
```

## ADS Input Contract

Place data in:

- `data/raw/ads/<dataset-id>/publications.jsonl` or `publications.json` (required)
- `data/raw/ads/<dataset-id>/references.jsonl` or `references.json` (optional)

Expected fields per record:

- `Bibcode`
- `Author` (list or string)
- `Title_en` (or `Title`)
- `Abstract_en` (or `Abstract`)
- `Year`
- `Affiliation` (also tolerates `Affilliation`)

## Outputs

Train run (`artifacts/metrics/<run_id>/`):

- `00_context.json` (`pipeline_scope: train`)
- `03_train_manifest.json`
- `04_clustering_config_used.json` (LSPO val eps resolution)
- `05_stage_metrics_<stage>.json` (`metric_scope: train`)
- `05_go_no_go_<stage>.json`
- optional `99_compare_train_to_baseline.json`

Infer run (`artifacts/metrics/<run_id>/`):

- `00_context.json` (`pipeline_scope: infer`)
- `01_input_summary.json`
- `02_preflight_infer.json` (memory/pair complexity estimate + feasibility)
- `03_pairs_qc.json`
- `04_cluster_qc.json`
- `04_source_export_qc.json` (source export mapping coverage)
- `05_stage_metrics_infer_ads.json` (`metric_scope: infer`)
- `05_go_no_go_infer_ads.json`
- optional `99_compare_infer_to_baseline.json`

Infer cluster exports (`artifacts/clusters/<run_id>/`):

- `ads_clusters_infer_ads.parquet`
- `publication_authors_infer_ads.parquet`

Source-mirrored infer exports (`artifacts/exports/<run_id>/`):

- `publications.disambiguated.jsonl`
- `references.disambiguated.jsonl` (if references input exists)

Mapping rule:

- `mention_id` is stable from normalized mentions.
- `author_uid` is added by clustering.
- Join key is always `mention_id`.
- In source-mirrored JSON outputs, original rows are preserved and `AuthorUID` is appended parallel to `Author`.

## Caching and Resume

- Default behavior is resume/reuse.
- Use `--force` to recompute for the same `run_id`.
- Cache tools:
  - `python3 -m src.cli cache doctor`
  - `python3 -m src.cli cache purge --target <...>` (dry-run by default, add `--yes` to apply)

## Legacy Policy

- Legacy artifacts/runs remain readable.
- New writes use the new split train/infer contracts only.
- Notebooks are legacy research tooling; CLI is the canonical product interface.
