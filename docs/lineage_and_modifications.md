# Project Lineage and Technical Modifications

This document explains the technical lineage of `ads-and`: what comes from
the NAND paper and imported NAND code snapshot, what was rebuilt in this
package, and how those changes affect reproducibility and reported metrics.
It is not a workflow runbook and it does not replace the citation and resource
links in `README.md`.

The document separates three layers:

- the NAND paper method
- the imported NAND snapshot at commit `af46945`
- the current `ads-and` package implementation

## Evidence Sources

Technical claims in this document are based on:

- the NAND paper cited in `README.md`
- the original imported snapshot available through
  `git show af46945:neural_name_dismabiguator-main/...`
- current package code under `src/author_name_disambiguation`
- the fixed bundle artifacts under
  `src/author_name_disambiguation/resources/model_bundles/fixed_model_baseline/bundle_v1`
- operational LSPO and ADS notes under `docs/` where they describe current
  repo workflows

## Upstream Method

NAND is an author name disambiguation method evaluated on LSPO. The paper
combines chars2vec name features with SPECTER title/abstract embeddings, learns
a contrastive pair-scoring representation, and clusters mentions within name
blocks with DBSCAN plus metadata constraints.

The NAND paper is the method and metric reference point. Its Table 3 reports
LSPO clustering with constraints at F1 95.93%, precision 96.15%, and recall
96.21%. This paper result should not be read as a guarantee that any imported
code snapshot is a complete, directly executable reproduction of the paper.

## Imported NAND Snapshot

This repository imported an upstream snapshot in commit `af46945` under
`neural_name_dismabiguator-main/`. That snapshot was later removed from the
tracked tree in commit `52194e9`.

The imported snapshot contained the NAND README, environment and Croissant
metadata, the paper PDF, and the scripts `AND_dataset_builder.py`,
`AND_nn_exp.py`, `AND_readdata_exp.py`, `experiment.py`, and `results.py`.
Together, those files describe a flat research-script workflow: build LSPO
pairs, train a PyTorch-Lightning pair-scoring model, and evaluate saved
checkpoints.

The snapshot is a lineage reference, not the source tree that is executed by
the current package. Its scripts use hardcoded local paths and environment
assumptions, and some references point to names that are not present in the
imported tree, such as `AND_nn_exp2` and `AND_readdata_exp2`. The current
package should therefore be described as a rebuilt package implementation in
the NAND/LSPO line, not as a direct packaging of a complete executable upstream
snapshot.

## Current Package Implementation

The public package surface is inference-focused:

- CLI: `ads-and infer`
- Python API: `author_name_disambiguation.disambiguate_sources`
- bundled Trained NAND Model: `full_20260218T111506Z_cli02681429`
- fixed bundle resource: `resources/model_bundles/fixed_model_baseline/bundle_v1`

The repo-only research surface supports an LSPO Gate Run, an ADS Full Candidate
Run, model-training experiments, and explicit model bundle export. Public ADS
inference reads ADS-shaped parquet inputs and writes disambiguated parquet
outputs with `AuthorUID` and `AuthorDisplayName`, plus assignment, entity,
cluster, summary, stage-metric, and go/no-go artifacts.

The Raw LSPO Source is not redistributed by this package. Repo workflows expect
the local Raw LSPO Source at `data/raw/lspo/LSPO_v1.parquet`.

## Technical Differences That Affect Reproducibility and Metrics

| Area | NAND paper / imported snapshot | Current package | Why it matters |
| --- | --- | --- | --- |
| Repository shape | Paper method plus flat scripts in `neural_name_dismabiguator-main/`. Users edit scripts, paths, embeddings, and checkpoint references. | Installable Python package under `src/author_name_disambiguation`, with public CLI/API and repo-only research commands. | Current results come from a rebuilt package workflow, not from running the imported scripts unchanged. |
| Data and pair construction | The snapshot builds within-block LSPO pairs from local files and then balances labels after pair construction. | Current pair building uses ORCID-aware splits, feasibility checks, split-balance/QC reports, `exclude_same_bibcode`, and manifests. | Pair construction defines the train/validation/test candidate space, so controlled split and balance behavior can change measured F1. |
| Pair-scoring model | The paper uses chars2vec plus SPECTER and contrastive learning. The imported `AND_nn_exp.py` encoder is an `818 -> 1024 -> 1024 -> 256` Lightning-style network. | The fixed package config uses `818 -> 1024 -> 256`; the current encoder normalizes output embeddings and training combines positive-pair InfoNCE with a negative-margin loss. | The learned similarity space is not text-identical to the imported script, so pairwise scores and downstream clusters can differ. |
| Thresholding | The imported evaluation code uses a ROC/G-mean-style threshold selection path. | Current training selects the cosine threshold by validation F1 sweep and records the selected threshold in the bundle manifest. | The threshold directly controls pair decisions and clustering edges. Different threshold policy can change precision, recall, and F1. |
| Clustering and constraints | The paper clusters within name blocks with DBSCAN and applies metadata constraints such as full-name conflicts and large year gaps. | Current clustering keeps that lineage but resolves eps through manifest-driven config, records sweep metadata, and applies explicit hard/soft constraint settings. | The clustering policy is reproducible from package artifacts, but it is not merely an implicit rerun of the original scripts. |
| Runtime and export | The snapshot evaluates manually selected checkpoints through local scripts. | Public inference resolves the bundled Trained NAND Model, writes ADS parquet outputs, and emits reports/manifests for inspection. | These changes mainly affect packaging, auditability, and ADS usability; they should not be described as proof of quality gains by themselves. |

## Metric Provenance

Two metric references are relevant and should not be collapsed:

- NAND paper Table 3 reports LSPO clustering with constraints at F1 95.93%,
  precision 96.15%, and recall 96.21%.
- `README.md` reports the current package LSPO clustering row at F1 97.02%,
  precision 96.36%, and recall 97.70%.

The tracked fixed bundle contains related but not identical metric artifacts.
`bundle_manifest.json` records the source model run
`full_20260218T111506Z_cli02681429`, selected eps `0.35`, threshold `0.502`,
and pairwise `best_test_f1 = 0.976252414576076`. `clustering_resolved.json`
records the validation eps-sweep selection, including F1
`0.9737836575953416` at eps `0.35`.

The complete LSPO clustering report behind the README row is redistributed as a
small repo-level reproduction artifact, not inside the public package wheel:
`artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__chars_cpu_20260407_v1.json`.
It evaluates seeds 1 through 5 and reports, for DBSCAN with constraints,
F1 `0.9702453597377284`, precision `0.9635797859580881`, and recall
`0.9770145661803212`. The matching seed checkpoints are tracked under
`artifacts/checkpoints/full_20260218T111506Z_cli02681429/`. Raw LSPO is not
redistributed and must be supplied by the user from the original Zenodo release.

The README clustering row should therefore be treated as the repo-reproducible
LSPO clustering result for this implementation. Without a tracked ablation
study, the higher package metrics should be described as consistent with the
technical differences above, not as caused by any single change.

## What This Document Does Not Claim

- It does not claim that the current code is byte-identical to the paper code.
- It does not claim that the bundled weights are upstream weights.
- It does not claim that the Raw LSPO Source is redistributed with the package.
- It does not claim that one implementation change alone explains the F1
  difference from the paper.

## License and Attribution

This repository is distributed under BSD-3-Clause.

The imported NAND README states BSD-3-Clause licensing for the upstream
project. LSPO is attributed through its Zenodo record and remains separately
licensed by its dataset terms, reported there as CC BY 4.0. Users should cite
the software entry in `CITATION.cff`; cite the original NAND paper and LSPO
dataset separately when discussing the underlying method or benchmark.
