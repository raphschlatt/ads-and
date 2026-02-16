# Runbook: NAND Best Path

## Goal

Reproduce NAND best path on LSPO, then run zero-shot disambiguation on ADS mentions.

## Best Path Settings

- Representation: `Chars2Vec + SPECTER`
- Loss: `InfoNCE + negative margin (explicit train negatives)`
- Network: `818 -> 1024 -> 256` with SELU + dropout 0.2
- Learning rate: `1e-5`
- Batch size: `2048` (reduce if VRAM-limited)
- Seeds: `1..5`
- Split: `60/20/20` (`split_assignment` in run config)
- Pair protocol: exclude same-publication pairs (`exclude_same_bibcode: true`)
- Clustering: DBSCAN + constraints, `eps_mode: val_sweep` in `0.20..0.50`

## Execution Order

1. Setup + config
2. Data prep + subset creation
3. Embeddings + pair building (LSPO)
4. Training
5. ADS inference + clustering
6. Stage report + gate check

## Full Run Gate

Proceed to `full` only if:

1. E2E stage run passes from fresh kernel
2. Determinism checks pass for subset and pair manifests
3. Schema validations pass
4. `mention_id -> author_uid` uniqueness is 1-to-1
5. LSPO metrics are in expected range for current stage
6. `lspo_pairwise_f1` is test-F1 (val-F1 is diagnostic only)
