# Runbook: NAND Best Path

## Goal

Reproduce NAND best path on LSPO, then run zero-shot disambiguation on ADS mentions.

## Best Path Settings

- Representation: `Chars2Vec + SPECTER`
- Loss: `InfoNCE`
- Network: `818 -> 1024 -> 256` with SELU + dropout 0.2
- Learning rate: `1e-5`
- Batch size: `2048` (reduce if VRAM-limited)
- Seeds: `1..5`
- Clustering: DBSCAN + lightweight constraints

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
