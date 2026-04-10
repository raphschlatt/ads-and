# Project Lineage and Modifications

This document records how `ads-and` relates to the NAND research code line and
what this repository changes. It is repo documentation, not a replacement for
the citation and resource links in `README.md`.

## Upstream References

- Original NAND repository: <https://github.com/deepthought-initiative/neural_name_dismabiguator>
- NAND paper: Vicente Amado Olivo et al. 2025, PASP 137, 124503, <https://doi.org/10.1088/1538-3873/ae1e2d>
- LSPO dataset: <https://doi.org/10.5281/zenodo.11489161>
- This package: `ads-and`, import path `author_name_disambiguation`

## Imported NAND Snapshot

This repository imported an upstream snapshot in commit `af46945` under
`neural_name_dismabiguator-main/`. That snapshot was later removed from the
tracked tree in commit `52194e9`.

The imported files were:

- `AND_dataset_builder.py`
- `AND_nn_exp.py`
- `AND_readdata_exp.py`
- `experiment.py`
- `results.py`
- `README.md`
- `environment.yml`
- `croissant.json`
- `Amado_Olivo_2025_PASP_137_124503.pdf`

The snapshot described a flat research-script workflow for LSPO-based Neural
Author Name Disambiguation. It built SPECTER plus chars2vec features, generated
within-block author-pair examples, trained a PyTorch-Lightning pair-scoring
network, and evaluated checkpoints with accuracy, precision, recall, F1, and
standard-error summaries.

The imported snapshot used hardcoded local paths in several scripts. Some
script references in that snapshot point to files or class names that were not
present in the imported tree, so this repository should not describe the
current package as a direct packaging of a complete executable snapshot.

## Current Package Surface

The public package surface is inference-focused:

- CLI: `ads-and infer`
- Python API: `author_name_disambiguation.disambiguate_sources`
- bundled Trained NAND Model: `full_20260218T111506Z_cli02681429`
- fixed model bundle resource: `resources/model_bundles/fixed_model_baseline/bundle_v1`

The repo-only research surface is exposed through:

- `python -m author_name_disambiguation_research`
- `run-cluster-test-report` and `quality-lspo` for an LSPO Gate Run
- `run-infer-sources` for an ADS Full Candidate Run
- `run-train-stage` and `train-lspo` for a model-training experiment
- `export-model-bundle` for explicit bundle creation

## What Changed

The current repository keeps the NAND/LSPO/SPECTER/chars2vec lineage but
rebuilds the workflow as a Python package and repo workspace:

- package layout under `src/author_name_disambiguation`
- public CLI and Python API for local ADS inference
- ADS parquet input support for publications and optional references
- output `AuthorUID` and `AuthorDisplayName` columns
- exported assignment, entity, cluster, summary, stage-metric, and go/no-go artifacts
- bundled fixed model resolution instead of user-supplied checkpoint paths for public inference
- repo-only LSPO Gate Run for inference-only experiments
- repo-only model-training experiment workflow with manifests and reports
- explicit `export-model-bundle` step after training
- runtime policy for CPU/GPU selection, SPECTER backend selection, chars2vec CPU behavior, clustering backend selection, and fallback diagnostics
- ADS inference baseline comparison and retention tooling

The model implementation is not text-identical to the imported Lightning
script. For example, the imported `AND_nn_exp.py` encoder used an
`818 -> 1024 -> 1024 -> 256` network, while the packaged fixed model config
uses `818 -> 1024 -> 256` with normalized output embeddings.

## Model and Data Provenance

The Raw LSPO Source is not redistributed by this package. The canonical local
path for repo workflows is `data/raw/lspo/LSPO_v1.parquet`.

The bundled Trained NAND Model is recorded by its source model run id,
`full_20260218T111506Z_cli02681429`, in the fixed bundle manifest. Do not
describe these bundled weights as upstream NAND weights unless that claim is
separately established.

The current ADS inference baseline is `bench_full_v22_fix2` and is documented
under `docs/baselines/`. That is an operational inference comparison target,
not the same concept as the trained model baseline.

## License and Attribution

This repository is distributed under BSD-3-Clause.

The original NAND repository README states BSD-3-Clause licensing for the
upstream project. LSPO is attributed to its Zenodo record and remains separately
licensed by its dataset terms, reported there as CC BY 4.0. Users should cite
the software entry in `CITATION.cff` and the NAND paper listed in `README.md`.
