# Project Lineage and Technical Modifications

This document explains where `ads-and` comes from, what the original NAND work
does, and what this package changed to make the method usable as a packaged ADS
inference tool. It is intentionally not a runbook. Use `README.md`,
`docs/inference_workflow.md`, and `docs/training_workflow.md` for commands.

The short version:

- NAND is the scientific method: author name disambiguation as learned
  similarity plus clustering inside name blocks.
- LSPO is the benchmark and training/evaluation source: NASA/ADS publications
  linked to ORCID identities.
- The upstream NAND repository is a research-script implementation, not the
  executable package surface used by `ads-and`.
- `ads-and` keeps the NAND/LSPO method lineage but rebuilds ingestion,
  training/evaluation plumbing, model bundling, inference, reporting, and
  runtime behavior around an installable Python package.
- The higher LSPO row reported by `ads-and` is repo-reproducible from tracked
  artifacts, but it should not be attributed to one isolated change without an
  ablation study.

## Evidence Sources

| Source | What it establishes |
| --- | --- |
| NAND paper: <https://doi.org/10.1088/1538-3873/ae1e2d> and arXiv page <https://arxiv.org/abs/2511.10722> | Method, LSPO construction, pairwise and clustering metrics, software/hardware context. |
| Original NAND repository: <https://github.com/deepthought-initiative/neural_name_dismabiguator> | Public upstream code shape: flat scripts for pair building, training, and evaluation. |
| LSPO Zenodo record: <https://doi.org/10.5281/zenodo.11489161> | Dataset identity, file release, license, and high-level dataset description. |
| Imported local snapshot: `git show af46945:neural_name_dismabiguator-main/...` | The exact upstream-style files once imported into this repository. |
| Current package source: `src/author_name_disambiguation/` | The public package and its internal implementation. |
| Current research surface: `author_name_disambiguation_research/` plus excluded repo modules | Repo-only training, quality, and benchmark workflows. |
| Fixed model bundle: `src/author_name_disambiguation/resources/model_bundles/fixed_model_baseline/bundle_v1/` | The public inference bundle shipped with the package. |
| LSPO metric artifact: `artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__chars_cpu_20260407_v1.json` | The five-seed LSPO clustering report behind the README metric row. |

## What NAND Does

NAND addresses author name disambiguation (AND): given publication metadata and
an author mention on each publication, infer which mentions belong to the same
real researcher.

Example problem:

```text
"Smith, J." on paper A
"Smith, Jane" on paper B
"Smith, John" on paper C
```

All three mentions may fall into the same name block, but they should not
necessarily become one identity. The method must decide whether the publication
context and the name evidence support a merge.

The NAND pipeline has five conceptual stages:

1. Build author mentions.

   A mention is one author occurrence on one publication. One paper with ten
   authors creates ten mentions.

2. Block by name.

   The search space is reduced by grouping mentions with the same first initial
   and surname, for example `j.smith`. Clustering happens inside a block, not
   across unrelated names.

3. Embed each mention.

   NAND represents the author name with chars2vec and the publication content
   with SPECTER title/abstract embeddings. The chars2vec plus SPECTER baseline
   uses a 50-dimensional name vector and a 768-dimensional text vector, giving
   an 818-dimensional input vector.

4. Learn a similarity space.

   A neural encoder maps each mention vector into a lower-dimensional embedding.
   Mentions by the same researcher should be close; mentions by different
   researchers should be farther apart.

5. Cluster inside each block.

   Pairwise similarities are converted to distances. DBSCAN, plus name and year
   constraints, partitions the block into inferred researcher identities.

The original NAND paper frames this as a contrastive, zero-shot similarity
learning approach: train once on ORCID-linked LSPO pairs, then apply the learned
similarity model to unseen name blocks.

## LSPO and the Original Evaluation Target

LSPO is the original benchmark behind NAND. The Zenodo release describes it as
553,496 NASA/ADS publications linked to 125,486 unique researchers via ORCID,
with metadata fields including ORCID, author name, affiliation, title,
abstract, and name block.

The NAND paper reports two relevant LSPO result levels:

| Evaluation | Result |
| --- | --- |
| Pairwise similarity, chars2vec + SPECTER + InfoNCE | F1 95.94% on balanced hard-positive/hard-negative pairs. |
| Clustering, DBSCAN + constraints | F1 95.93%, precision 96.15%, recall 96.21%. |

The clustering row is the most relevant comparison point for `ads-and`, because
the public package returns clustered researcher identities, not just pairwise
same-author scores.

## Original NAND Repository and Imported Snapshot

The upstream repository is a compact research-code release. Its README describes:

- `AND_dataset_builder.py` for pair construction
- `AND_nn_exp.py` for the model architecture
- `AND_readdata_exp.py` for the data module
- `experiment.py` for running training across seeds
- `results.py` for evaluating checkpoints
- a conda environment

The same upstream-style files were imported into this repository in commit
`af46945` under `neural_name_dismabiguator-main/` and later removed from the
tracked tree in commit `52194e9`.

That snapshot is useful lineage evidence, but it is not the package
implementation:

- It is script-oriented rather than package-oriented.
- It contains hardcoded local paths for embeddings, checkpoints, and threshold
  files.
- Its README says users must update checkpoint files and representation-specific
  paths for experiments.
- `experiment.py` imports `AND_nn_exp2` and `AND_readdata_exp2`, names that are
  not present in the imported tree.
- The model script defines the chars2vec + SPECTER encoder as
  `818 -> 1024 -> 1024 -> 256`, while the current fixed package config uses
  `818 -> 1024 -> 256`.

So `ads-and` should be described as a rebuilt package implementation in the
NAND/LSPO line, not as a direct execution wrapper around a complete upstream
snapshot.

## What `ads-and` Adds

### Public Package Surface

The public package path is inference-focused:

- CLI: `ads-and infer`
- Python API: `author_name_disambiguation.disambiguate_sources`
- optional Modal backend: `--backend modal`
- optional cost lookup for Modal runs: `ads-and cost`

Public inference takes ADS-shaped publication parquet input, optionally
reference parquet input, and writes:

- `publications_disambiguated.parquet`
- `references_disambiguated.parquet`, when references are provided
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `summary.json`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

The disambiguated source parquets preserve the input columns and append
parallel list columns:

```text
AuthorUID          ["ads_run::s.cole::1", "ads_run::c.lacey::0"]
AuthorDisplayName  ["Cole, Shaun", "Lacey, C. G."]
```

The lists are in the same order as the input `Author` list.

### ADS Ingestion and Source Outputs

The original NAND work is LSPO-centered. `ads-and` adds a production ADS
source path:

1. Read publication and reference parquets with minimal column projection.
2. Normalize ADS column variants such as `Bibcode`/`bibcode`,
   `Title_en`/`Title`, and `Abstract_en`/`Abstract`.
3. Deduplicate publication/reference records by bibcode while preserving the
   best available metadata.
4. Explode records into author mentions.
5. Infer clusters.
6. Project cluster assignments back onto the original source rows.

This is a major usability difference: the user does not have to manually build
LSPO-style pairs, edit checkpoint paths, or post-process cluster labels into ADS
source files.

### Fixed Model Bundle

The package ships one fixed inference bundle:

```text
full_20260218T111506Z_cli02681429
```

The bundle resource is:

```text
src/author_name_disambiguation/resources/model_bundles/fixed_model_baseline/bundle_v1/
```

It contains:

- `checkpoint.pt`
- `model_config.yaml`
- `clustering_resolved.json`
- `bundle_manifest.json`

Important recorded bundle values:

| Field | Value |
| --- | ---: |
| `source_model_run_id` | `full_20260218T111506Z_cli02681429` |
| `checkpoint_hash` | `5042119c06e2` |
| `best_threshold` | `0.502` |
| `selected_eps` | `0.35` |
| `precision_mode` | `fp32` |
| `best_test_f1` in bundle manifest | `0.976252414576076` |

The bundle manifest's `best_test_f1` is a pairwise test metric for the selected
checkpoint. It is not the same object as the five-seed LSPO clustering row in
the README.

## Technical Modifications

| Area | Original paper / snapshot | Current package | Why it matters |
| --- | --- | --- | --- |
| Repository shape | Paper method plus flat scripts. | Installable package under `src/author_name_disambiguation`, with a small public CLI/API and a separate repo-only research surface. | Users can run inference without editing research scripts. |
| Public scope | LSPO research workflow. | ADS parquet inference with source-preserving outputs and stable IDs. | Turns the method into a usable ADS disambiguation tool. |
| Data ingestion | Local LSPO files and manually prepared embeddings. | ADS parquet normalization, publication/reference deduplication, mention fanout, and source-output projection. | Makes input/output behavior explicit and reproducible. |
| Splits | Paper describes ORCID-level splits before pair construction. Snapshot scripts are manual and path-bound. | ORCID-aware splits, split feasibility checks, split-balance metadata, deterministic seeding, and manifests. | Reduces leakage risk and makes train/eval conditions auditable. |
| Pair construction | Within-block pairs, then class balancing. | Within-block pairs with `exclude_same_bibcode`, optional block caps, train balancing, chunked parquet output, and CPU sharding. | Pair construction is both quality-sensitive and runtime-sensitive. |
| Encoder | Snapshot chars2vec + SPECTER script uses `818 -> 1024 -> 1024 -> 256`. | Fixed config uses `818 -> 1024 -> 256`; output embeddings are L2-normalized. | Pair scores are not byte-identical to the imported script. |
| Training objective | Paper evaluates contrastive losses; snapshot contains cosine, NCE, and triplet paths. | Repo training combines positive-pair InfoNCE with an explicit negative-margin loss. | This changes the learned similarity geometry and can affect precision/recall. |
| Threshold selection | Paper describes ROC-based validation thresholding; snapshot stores threshold arrays in local paths. | Current training sweeps cosine thresholds on validation F1 and records threshold source/status in manifests. | The threshold directly controls pair edges and downstream clusters. |
| Clustering eps | Paper tunes DBSCAN behavior empirically; snapshot has less packaged provenance. | Bundle records validation eps sweep and selected `eps=0.35`. | The clustering setting is recoverable from package artifacts. |
| Constraints | Paper uses name and temporal constraints. | Current bundle uses hard full-name conflict constraints and soft year-gap constraints with recorded distances. | These constraints are part of the quality contract, not just runtime behavior. |
| Runtime policy | Local scripts and HPC-oriented experiments. | Explicit local CPU/GPU policy, optional ONNX CPU backend, Modal remote GPU backend, preflight checks, progress, stage metrics, and go/no-go reports. | Runtime behavior is visible and debuggable. |
| Clustering runtime | DBSCAN over per-block precomputed distances. | For `metric=precomputed` and `min_samples=1`, the package can use an exact connected-components graph path after applying the same edge/constraint logic. | This is a large runtime improvement but still quality-sensitive and validated through LSPO/ADS workflows. |
| Export | Evaluation metrics from checkpoints. | Source-preserving parquets, row-level assignments, entity table, cluster table, and summary JSON. | Makes results inspectable outside the research pipeline. |

## Metric Provenance

Two metric rows should be kept separate:

| Source | LSPO clustering setting | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: |
| NAND paper | DBSCAN + constraints | 95.93% | 96.15% | 96.21% |
| `ads-and` tracked report | DBSCAN + constraints, five seeds | 97.02% | 96.36% | 97.70% |

The `ads-and` row comes from:

```text
artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__chars_cpu_20260407_v1.json
```

That report evaluates seeds 1 through 5. For the constrained variant it records:

```text
f1_mean        0.9702453597377284
precision_mean 0.9635797859580881
recall_mean    0.9770145661803212
```

The matching seed checkpoints are tracked under:

```text
artifacts/checkpoints/full_20260218T111506Z_cli02681429/
```

Raw LSPO is not redistributed. To rerun the quality workflow, the user must
supply LSPO from Zenodo locally.

### Why the Metrics Can Differ

The package result can differ from the NAND paper because the current
implementation is not a byte-identical execution of the paper repository. The
most plausible quality-relevant differences are:

- controlled ORCID split and pair-building machinery
- same-publication pair exclusion
- current encoder shape and normalized outputs
- combined InfoNCE plus negative-margin training objective
- validation-F1 threshold selection
- manifest-resolved DBSCAN eps selection
- explicit hard/soft constraint settings

These differences are enough to make metric differences expected. However, the
repository does not contain a full ablation study that isolates each change.
Therefore the correct claim is:

```text
The current package reports a higher five-seed LSPO clustering result under its
tracked implementation and quality workflow.
```

The incorrect claim would be:

```text
One specific implementation change explains the full F1 improvement.
```

## Runtime Provenance

The original paper reports research compute for generating embeddings and
training NAND models. That is not directly comparable to public `ads-and infer`,
because public inference uses a fixed bundled checkpoint and does not train.

The current package improves runtime and operability mainly by changing the
inference path:

- no training required for public inference
- vectorized parquet loading and minimal projection
- reusable embedding caches
- SPECTER batching and device-aware precision
- chars2vec CPU default in the product path
- preencoded mention embeddings for pair scoring
- chunked parquet pair-score output
- exact connected-components clustering for the `min_samples=1` threshold graph
- export frame reuse instead of rereading source parquets
- Modal remote GPU execution for users without local GPU hardware

One tracked ADS Modal L4 run provides an operational scale reference:

```text
artifact: artifacts/exports/ads_prod_current_modal_full_l4_v1/
source rows: 183,516 publications + 441,957 references
author mentions: 1,322,725
clusters: 231,352
stage total: 1,445.5 seconds
actual Modal cost: $0.53310679
```

This supports the README rule of thumb of roughly 2.5 seconds and $0.00085 per
1,000 ADS source records on that L4-backed run. Exact cost remains a Modal
billing lookup, not a static package guarantee.

Runtime changes are not all "runtime-only" from a quality perspective. In
particular, chars2vec execution, pair scoring, threshold graph construction, and
constraint application can change outputs if altered. The repository therefore
treats them as quality-sensitive and validates them through LSPO quality gates
before using an ADS full inference run as the deciding benchmark.

## Reproducibility Boundaries

The current package makes these commitments:

- The public package includes the fixed inference bundle.
- README LSPO metrics are backed by tracked repo artifacts.
- The repo-only `quality-lspo` workflow can reproduce the LSPO report when Raw
  LSPO is supplied locally.
- Public `ads-and infer` uses the fixed bundle and does not require the
  upstream research scripts.

It does not claim:

- that the package code is byte-identical to the paper code
- that the bundled weights are upstream weights
- that Raw LSPO is redistributed with the package
- that runtime optimizations alone explain quality gains
- that optional ONNX CPU is universally faster on every host
- that one single change explains the full F1 difference from the paper

## License and Attribution

This repository is distributed under BSD-3-Clause.

The original NAND repository README also states BSD-3-Clause licensing for the
upstream project. LSPO is separately distributed through Zenodo under CC BY 4.0.

Cite `ads-and` as software via `CITATION.cff` or the Zenodo concept DOI listed
in `README.md`. Cite the original NAND paper when discussing the underlying
method, and cite LSPO separately when discussing the benchmark or dataset.
