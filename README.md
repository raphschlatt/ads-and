# ads-and

[![PyPI](https://img.shields.io/pypi/v/ads_and.svg)](https://pypi.org/project/ads-and/)
[![Python](https://img.shields.io/pypi/pyversions/ads_and.svg)](https://pypi.org/project/ads-and/)
[![License](https://img.shields.io/pypi/l/ads_and.svg)](https://github.com/raphschlatt/ads-and/blob/main/LICENSE)

`ads-and` is a Python package for author name disambiguation (AND) on [SAO/NASA ADS](https://ui.adsabs.harvard.edu/) records. Given publications and optionally references in ADS parquet format, it assigns stable author identifiers and writes disambiguated outputs.

The bundled model is a packaged and slightly refined version of [NAND](https://github.com/deepthought-initiative/neural_name_dismabiguator) (Neural Author Name Disambiguator), described in [Amado Olivo et al. 2025](https://doi.org/10.1088/1538-3873/ae1e2d). NAND was trained and evaluated on [LSPO](https://doi.org/10.5281/zenodo.11489161), a large-scale physics and astronomy AND benchmark built from ~553k NASA/ADS publications linked to ORCID identities (~125k researchers). The model ships inside the package, no external bundle is required.

This implementation was re-evaluated on LSPO under a five-seed protocol.
Clustering performance on LSPO (with constraints enabled):

| | F1 | Precision | Recall |
|---|---|---|---|
| NAND — Amado Olivo et al. 2025 | 95.93% | 96.15% | 96.21% |
| `ads-and` (this package) | **97.02%** | **96.36%** | **97.70%** |

Python import path: `author_name_disambiguation`

## Install

Use [uv](https://docs.astral.sh/uv/). Requires Python 3.12.

```powershell
uv pip install ads-and
```

If you don't have a GPU: optional ONNX CPU backend, which may be faster
depending on host and workload:

```powershell
uv pip install "ads-and[cpu_onnx]"
```

Optional [Modal](https://modal.com/) backend (you need a modal account):

```powershell
uv pip install "ads-and[modal]"
```

## Usage

**CLI**

```powershell
ads-and infer `
  --publications-path path/to/publications.parquet `
  --references-path path/to/references.parquet `
  --output-dir path/to/output-dir `
  --runtime auto
```

Add `--json` for a machine-readable run summary on stdout.

`--runtime` options: `auto` (GPU if CUDA is available, else CPU), `gpu`, `cpu`.
Advanced infer flags such as `--infer-stage`, `--dataset-id`, and
`--modal-gpu` are documented in
[`docs/inference_workflow.md`](docs/inference_workflow.md).

Modal uses the same command surface with [Modal](https://modal.com/) as a
managed remote GPU backend (you need a modal account):

```powershell
ads-and infer `
  --publications-path path/to/publications.parquet `
  --references-path path/to/references.parquet `
  --output-dir path/to/output-dir `
  --backend modal `
  --runtime gpu `
  --modal-gpu l4
```

Current repo Modal config is `--backend modal --runtime gpu --modal-gpu l4`. The
local client uploads the ADS parquet inputs, Modal runs the same bundled infer
workflow remotely, and the finished outputs are copied back into `output-dir`.
Current `L4` rule of thumb: about `$0.00085` and `~2.5s` per `1,000` ADS entries.
Configure
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in your environment or a repo-root
`.env` before using `--backend modal`.

Exact Modal costs are a separate official lookup:

```powershell
ads-and cost --output-dir path/to/output-dir
```

This is a follow-up lookup after the run, once the billing window has closed.

**Python**

Local CPU/GPU:

```python
from author_name_disambiguation import disambiguate_sources

result = disambiguate_sources(
    publications_path="path/to/publications.parquet",
    references_path="path/to/references.parquet",
    output_dir="path/to/output-dir",
    runtime="auto",
)

print(result.publications_disambiguated_path)
print(result.summary_path)
```

Modal:

```python
from author_name_disambiguation import disambiguate_sources, resolve_modal_cost

modal_result = disambiguate_sources(
    publications_path="path/to/publications.parquet",
    references_path="path/to/references.parquet",
    output_dir="path/to/output-dir",
    backend="modal",
    runtime="gpu",
    modal_gpu="l4",
)

# later, after the billing interval closes
cost_result = resolve_modal_cost("path/to/output-dir")
```

## Input schema

`--publications-path` is required. `--references-path` is optional.

| Column | Required | Type | Example |
| --- | --- | --- | --- |
| `Bibcode` | **yes** | `str` | `"2000MNRAS.319..168C"` |
| `Author` | **yes** | `list[str]` or semicolon-delimited `str` | `["Cole, Shaun", "Lacey, Cedric G."]` |
| `Title_en` or `Title` | no — but strongly recommended | `str` | `"Galaxy luminosity functions in..."` |
| `Abstract_en` or `Abstract` | no — but strongly recommended | `str` | `"We model the galaxy population..."` |
| `Affiliation` | no | `str` (ADS format) or `list[str]` (per-author) | `"AA(Durham Univ, Dept of Physics); AB(...)"` |
| `Year` | no | `int` | `2000` |

Records missing `Bibcode` or `Author` are skipped. Records missing both `Title` and `Abstract` will be processed but with meaningfully reduced disambiguation quality, since the model relies heavily on textual context to distinguish authors.

## Output

All files are written under `output_dir`:

| File | Contents |
| --- | --- |
| `publications_disambiguated.parquet` | input columns + `AuthorUID`, `AuthorDisplayName` |
| `references_disambiguated.parquet` | same, for references (only when references are provided) |
| `source_author_assignments.parquet` | row-level author-to-entity assignments |
| `author_entities.parquet` | inferred author entities |
| `mention_clusters.parquet` | mention-to-cluster mapping |
| `summary.json` | high-level run summary |
| `05_stage_metrics_infer_sources.json` | diagnostic per-stage runtime and validation metrics |
| `05_go_no_go_infer_sources.json` | diagnostic run validation summary |

The two disambiguated parquets preserve all input columns and append:

| Column | Type | Example |
| --- | --- | --- |
| `AuthorUID` | `list[str]` | `["ads_run::s.cole::1", "ads_run::c.lacey::0", "ads_run::c.baugh::0"]` |
| `AuthorDisplayName` | `list[str]` | `["Cole, Shaun", "Lacey, C. G.", "Baugh, C. M."]` |

Both columns are parallel lists in the same order as the input `Author` column. Each UID is stable across runs for the same registry. Each author entity gets exactly one display name — the most frequently occurring form of their name in the data (could be full-name or abbreviated depending on the entity). The same UID always carries the same display name string.

## Reproducibility

The bundled inference model is the selected fixed model from
`full_20260218T111506Z_cli02681429`. The five-seed LSPO result above is backed
by tracked repo-level artifacts under `artifacts/`, including the five seed
checkpoints and the canonical clustering report. Raw LSPO is not redistributed;
download it separately from Zenodo to rerun the quality workflow.

See [Training workflow](https://github.com/raphschlatt/ads-and/blob/main/docs/training_workflow.md)
for the exact LSPO reproduction and release-gate commands.

### Further Details

- [Inference workflow](https://github.com/raphschlatt/ads-and/blob/main/docs/inference_workflow.md)
- [LSPO reproduction and training workflow](https://github.com/raphschlatt/ads-and/blob/main/docs/training_workflow.md)
- [Project lineage and modifications](https://github.com/raphschlatt/ads-and/blob/main/docs/lineage_and_modifications.md)

## Citation

Cite `ads-and` as software via [`CITATION.cff`](CITATION.cff). Cite the original NAND paper if you discuss the underlying method or baseline:

> Vicente Amado Olivo, Wolfgang Kerzendorf, Bangjing Lu, Joshua V. Shields, Andreas Flörs, and Nutan Chen (2025). *Practical Author Name Disambiguation under Metadata Constraints: A Contrastive Learning Approach for Astronomy Literature.* Publications of the Astronomical Society of the Pacific, 137(12), 124503. <https://doi.org/10.1088/1538-3873/ae1e2d>

And cite LSPO separately if you discuss the benchmark or dataset:

> Vicente Amado Olivo (2024). *LSPO: A Large-Scale Physics ORCiD-Linked Dataset for Author Name Disambiguation.* Zenodo, Version 1. <https://doi.org/10.5281/zenodo.11489161>

Resources:
- Original NAND repository: <https://github.com/deepthought-initiative/neural_name_dismabiguator>
- Original NAND paper: <https://doi.org/10.1088/1538-3873/ae1e2d>
- LSPO dataset: <https://doi.org/10.5281/zenodo.11489161>