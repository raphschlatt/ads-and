# Runbook: NAND Best Path (Train/Infer Split)

## Goal

1. Train and benchmark NAND on LSPO (`run-train-stage`).
2. Export/deploy model bundle.
3. Disambiguate ADS datasets (`run-infer-ads`).

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

Infer (run id source):

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-run-id smoke_2026... \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Infer (bundle source):

```bash
python3 -m src.cli run-infer-ads \
  --dataset-id my_ads_2026 \
  --model-bundle artifacts/models/smoke_2026.../bundle_v1 \
  --paths-config configs/paths.local.yaml \
  --device auto
```

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
