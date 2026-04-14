# ads-and

`ads-and` is a Python package for author name disambiguation (AND) on [SAO/NASA ADS](https://ui.adsabs.harvard.edu/) records. Given publications and optionally references in ADS parquet format, it assigns stable author identifiers and writes disambiguated outputs. It is scoped to the ADS column schema and is not a general-purpose AND toolkit for arbitrary metadata.

The bundled model is a packaged and slightly refined version of [NAND](https://github.com/deepthought-initiative/neural_name_dismabiguator) (Neural Author Name Disambiguator), described in [Amado Olivo et al. 2025](https://doi.org/10.1088/1538-3873/ae1e2d). NAND was trained and evaluated on [LSPO](https://doi.org/10.5281/zenodo.11489161), a large-scale physics and astronomy AND benchmark built from ~553k NASA/ADS publications linked to ORCID identities (~125k researchers). The model ships inside the package, no external bundle is required.

The bundled package was re-evaluated on the same LSPO benchmark under a reproducible five-seed protocol. Clustering performance on LSPO (with constraints enabled):

| | F1 | Precision | Recall |
|---|---|---|---|
| NAND — Amado Olivo et al. 2025 | 95.93% | 96.15% | 96.21% |
| `ads-and` (this package) | **97.02%** | **96.36%** | **97.70%** |

Python import path: `author_name_disambiguation`

## Install

Use [uv](https://docs.astral.sh/uv/). Requires Python ≥ 3.11.

```powershell
uv pip install ads-and
```

If you don't have a GPU: Optional faster CPU inference via ONNX (still much slower than GPU):

```powershell
uv pip install "ads-and[cpu_onnx]"
```

Optional Modal backend:

```powershell
uv pip install "ads-and[modal]"
```

## Usage

**CLI**

```powershell
ads-and infer `
  --publications-path data/ads/publications.parquet `
  --references-path data/ads/references.parquet `
  --output-dir outputs/ads_run `
  --runtime auto
```

Add `--json` for a machine-readable run summary on stdout.

`--runtime` options: `auto` (GPU if CUDA is available, else CPU), `gpu`, `cpu`.
Advanced infer flags such as `--infer-stage`, `--dataset-id`, and
`--modal-gpu` are documented in
[`docs/inference_workflow.md`](docs/inference_workflow.md).

Modal uses the same command surface with an explicit backend switch:

```powershell
ads-and infer `
  --publications-path data/ads/publications.parquet `
  --references-path data/ads/references.parquet `
  --output-dir outputs/ads_run_modal `
  --backend modal `
  --runtime auto
```

Modal runs the same ADS inference path remotely on managed GPU hardware. The
local client stages projected ADS parquet inputs to the remote job and copies
the finished outputs back into `output-dir`. Configure `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` in your environment or a repo-root `.env` before using
`--backend modal`.

Exact Modal costs are a separate official lookup:

```powershell
ads-and cost --output-dir outputs/ads_run_modal
```

This is a follow-up lookup after the run, once the billing window has closed.

**Python**

```python
from author_name_disambiguation import disambiguate_sources, resolve_modal_cost

result = disambiguate_sources(
    publications_path="data/ads/publications.parquet",
    references_path="data/ads/references.parquet",
    output_dir="outputs/ads_run",
    runtime="auto",
)

print(result.publications_disambiguated_path)
print(result.summary_path)

modal_result = disambiguate_sources(
    publications_path="data/ads/publications.parquet",
    references_path="data/ads/references.parquet",
    output_dir="outputs/ads_run_modal",
    backend="modal",
    runtime="auto",
)

# later, after the billing interval closes
cost_result = resolve_modal_cost("outputs/ads_run_modal")
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

## Further Details

Inference is out of the box because the bundled fixed model ships inside the
package. Repo-only research workflows require user-supplied LSPO raw data
from the original source release; both parquet and HDF5 inputs are supported
for LSPO preparation and evaluation.

- [Inference workflow](https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/inference_workflow.md)
- [Training workflow](https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/training_workflow.md)
- [Project lineage and modifications](https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/lineage_and_modifications.md)

## Citation

If you use `ads-and`, cite the software entry in [`CITATION.cff`](CITATION.cff) and the underlying NAND paper:

Vicente Amado Olivo, Wolfgang Kerzendorf, Bangjing Lu, Joshua V. Shields, Andreas Flörs, and Nutan Chen (2025). *Practical Author Name Disambiguation under Metadata Constraints: A Contrastive Learning Approach for Astronomy Literature.* Publications of the Astronomical Society of the Pacific, 137(12), 124503. <https://doi.org/10.1088/1538-3873/ae1e2d>

Resources:
- NAND repository: <https://github.com/deepthought-initiative/neural_name_dismabiguator>
- LSPO dataset: <https://doi.org/10.5281/zenodo.11489161>
