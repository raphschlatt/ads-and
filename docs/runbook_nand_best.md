# Runbook: NAND Best Path (Train/Infer Split)

## Goal

1. Train and benchmark NAND on LSPO (`run-train-stage`).
2. Generate final LSPO test clustering report (`run-cluster-test-report`).
3. Export/deploy model bundle.
4. Disambiguate ADS datasets (`run-infer-ads`).

## Canonical Settings

- Representation: `Chars2Vec + SPECTER`
- Loss: `InfoNCE + negative margin`
- Network: `818 -> 1024 -> 256` (SELU + dropout)
- Split: `60/20/20`
- Pair rule: `exclude_same_bibcode: true`
- Clustering: DBSCAN `eps_mode: val_sweep`, range `0.20..0.50`
- Boundary diagnostics: `0.55..0.70` audit-only
- Precision for benchmark runs: `fp32`

## Commands

Train:

```bash
python3 -m src.cli run-train-stage \
  --run-stage smoke \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Export bundle:

```bash
python3 -m src.cli export-model-bundle \
  --model-run-id smoke_2026... \
  --paths-config configs/paths.local.yaml
```

Final clustering test report:

```bash
PYTHONPATH=. python3 -m src.cli run-cluster-test-report \
  --model-run-id smoke_2026... \
  --paths-config configs/paths.local.yaml \
  --device auto \
  --precision-mode fp32
```

Infer (run id source):

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-run-id smoke_2026... \
  --infer-stage full \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Infer (bundle source):

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-bundle artifacts/models/smoke_2026.../bundle_v1 \
  --infer-stage mini \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Infer stage presets:

- `full` (default): process full deduplicated ADS mentions.
- `smoke|mini|mid`: deterministic subset profiles from `configs/infer_runs/*.yaml`.

## Gate Expectations

Train scope checks:

- schema/determinism/run-id consistency
- split feasibility and negative coverage
- test-based `lspo_pairwise_f1`
- eps diagnostics (`boundary_hit`, `range_limited`)

Infer scope checks:

- mention coverage + UID uniqueness
- pair score range
- cluster quality rates
- eps diagnostics from model/source context

## Notes

- `run-stage` is deprecated and maps to train-only behavior.
- Legacy runs remain readable (read-only compatibility).
- `raw_lspo_h5` warning can be ignored when `data/raw/lspo/LSPO_v1.parquet` exists.

## Lean Baseline Anchor

Canonical baseline id:

- `full_20260218T111506Z_cli02681429`

Before comparing any new experiment, run:

```bash
PYTHONPATH=. python3 scripts/ops/check_baseline_integrity.py \
  --baseline-run-id full_20260218T111506Z_cli02681429
```

Pass criteria:

- required keep-set files exist
- `subset_cache_key_expected == subset_cache_key_computed`
- `06_clustering_test_report.json` has `status=ok` and seeds `[1,2,3,4,5]`

If the check reports missing files, restore only those missing paths from:

- `/home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved`

Targeted restore for the known shared baseline set:

```bash
mkdir -p data/cache/_shared/subsets data/cache/_shared/embeddings data/cache/_shared/pairs data/cache/_shared/eps_sweeps
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/subsets/lspo_mentions_full_seed11_targetfull_cfg0dbcdaf9_srcd52b159f766e.parquet data/cache/_shared/subsets/
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/embeddings/lspo_chars2vec_05757fec0582.npy data/cache/_shared/embeddings/
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/embeddings/lspo_specter_05757fec0582.npy data/cache/_shared/embeddings/
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/pairs/lspo_mentions_split_978ea2bd7512.parquet data/cache/_shared/pairs/
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/pairs/lspo_pairs_978ea2bd7512.parquet data/cache/_shared/pairs/
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/pairs/split_balance_978ea2bd7512.json data/cache/_shared/pairs/
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/pairs/pairs_qc_train_978ea2bd7512.json data/cache/_shared/pairs/
cp -a /home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved/data/_shared/eps_sweeps/eps_sweep_4f69281cae15.json data/cache/_shared/eps_sweeps/
```

## Quality Experiment Gate (`eps` buckets)

Use this cycle for isolated clustering-quality iteration without touching blocking logic or retraining:

```bash
# Candidate clustering report from existing baseline train-run
# --report-tag is mandatory when --cluster-config-override is used.
PYTHONPATH=. python3 -m src.cli run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml \
  --cluster-config-override configs/clustering/dbscan_paper_eps_buckets_v1.yaml \
  --report-tag epsbkt_v1 \
  --device auto \
  --precision-mode fp32
```

Hard gate decision (LSPO-only):

```bash
PYTHONPATH=. python3 scripts/ops/compare_cluster_test_reports.py \
  --baseline-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report.json \
  --candidate-report artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__epsbkt_v1.json \
  --variant dbscan_with_constraints \
  --min-delta-f1 0.0 \
  --max-precision-drop 0.001
```

Decision policy:

- exit `0` => promote candidate
- exit `1` => rollback to baseline

Freeze + Soft Rollback artifacts:

```bash
PYTHONPATH=. python3 scripts/ops/freeze_eps_experiments.py \
  --baseline-run-id full_20260218T111506Z_cli02681429
```

Expected outputs:

- `98_eps_experiments_manifest.json`
- `98_eps_experiments_manifest.md`
- `98_active_baseline.json`

Current status: `dbscan_paper_eps_buckets_v1.yaml`, `dbscan_paper_eps_buckets_v2.yaml`, `dbscan_paper_eps_buckets_v3.yaml` remain experiment-only and are not promoted.
