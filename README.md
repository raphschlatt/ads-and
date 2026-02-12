# Author Name Disambiguation (NAND-First, Notebook Interface)

Research-first setup to:

1. Reproduce NAND best path on LSPO (`Chars2Vec + SPECTER + InfoNCE`, `818 -> 1024 -> 256`).
2. Apply the trained model to ADS mentions.
3. Execute the workflow from notebook cells with visible intermediate QC.

## Notebook Calling Layer

Run these notebooks in order:

1. `notebooks/interface/00_setup_and_config.ipynb`
2. `notebooks/interface/01_data_and_subsets.ipynb`
3. `notebooks/interface/02_embeddings_and_pairs.ipynb`
4. `notebooks/interface/03_train_nand_best.ipynb`
5. `notebooks/interface/04_infer_cluster_ads.ipynb`
6. `notebooks/interface/05_run_report_and_go_no_go.ipynb`

## Stable IDs And Contracts

- `bibcode`: unique publication key
- `mention_id`: unique author mention key (`bibcode::author_idx`)
- `author_uid`: final disambiguated author ID for each mention

Contracts are documented in `docs/data_contracts.md`.

## Project Layout

- `src/`: backend modules (idempotent, notebook-callable)
- `configs/`: model, clustering, stage and path configs
- `data/raw|interim|processed|subsets/`: data flow and subset manifests
- `artifacts/`: embeddings, checkpoints, pair scores, clusters, metrics
- `notebooks/interface/`: execution interface
- `notebooks/reports/`: analysis/report notebooks
- `tests/`: unit/integration/e2e tests

## Stage Ladder (Gate Before Full)

- `smoke`: 1k mentions
- `mini`: 10k mentions
- `mid`: 100k mentions
- `full`: complete dataset

Each stage produces manifests and metrics under `artifacts/metrics/<run_id>/`.

## Optional CLI

The notebook layer is primary. A matching CLI exists for automation:

```bash
python3 -m src.cli -h
```

## Environment

Install with your preferred environment manager; a minimal pip list is in:

- `requirements-research.txt`
- Path profile defaults to `configs/paths.local.yaml` (switch to `configs/paths.colab.yaml` if needed).

## Notes

- Research setup, not production hardening.
- `smoke/mini` are intended for fast validation before expensive full runs.
- For real reproduction/training you need `torch`, `transformers`, and GPU runtime.
