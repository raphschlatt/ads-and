# ads-and

`ads-and` is a Python package for author name disambiguation (AND) on [SAO/NASA ADS](https://ui.adsabs.harvard.edu/) records. Given publications and optionally references in ADS parquet format, it assigns stable author identifiers and writes disambiguated outputs. It is scoped to the ADS column schema and is not a general-purpose AND toolkit for arbitrary metadata.

The bundled model is a packaged and slightly refined version of [NAND](https://github.com/deepthought-initiative/neural_name_dismabiguator) (Neural Author Name Disambiguator), described in [Amado Olivo et al. 2025](https://doi.org/10.1088/1538-3873/ae1e2d). NAND was trained and evaluated on [LSPO](https://doi.org/10.5281/zenodo.11489161), a large-scale physics and astronomy AND benchmark built from ~553k NASA/ADS publications linked to ORCID identities (~125k researchers). The model ships inside the package — no external bundle required.

The bundled package was re-evaluated on the same LSPO benchmark under a reproducible five-seed protocol. Clustering performance on LSPO (with constraints enabled):

| | F1 | Precision | Recall |
|---|---|---|---|
| NAND — Amado Olivo et al. 2025 | 95.93% | 96.15% | 96.21% |
| `ads-and` (this package) | **97.02%** | **96.36%** | **97.70%** |

Python import path: `author_name_disambiguation`

## Install

Use [uv](https://docs.astral.sh/uv/). Requires Python ≥ 3.11.

```bash
uv pip install ads-and
```

If you don't have a GPU: Optional faster CPU inference via ONNX (still much slower than GPU):

```bash
uv pip install "ads-and[cpu_onnx]"
```

## Usage

**CLI**

```bash
ads-and infer \
  --publications-path data/ads/publications.parquet \
  --references-path data/ads/references.parquet \
  --output-dir outputs/ads_run \
  --runtime auto
```

Add `--json` for a machine-readable run summary on stdout.

`--runtime` options: `auto` (GPU if CUDA is available, else CPU), `gpu`, `cpu`.

**Python**

```python
from author_name_disambiguation import disambiguate_sources

result = disambiguate_sources(
    publications_path="data/ads/publications.parquet",
    references_path="data/ads/references.parquet",
    output_dir="outputs/ads_run",
    runtime="auto",
)

print(result.publications_disambiguated_path)
print(result.summary_path)
```

## Input schema

`--publications-path` is required. `--references-path` is optional.

| Column | Required | Notes |
| --- | --- | --- |
| `Bibcode` | **yes** | ADS source identifier |
| `Author` | **yes** | author name list |
| `Title_en` or `Title` | no — but strongly recommended | title text |
| `Abstract_en` or `Abstract` | no — but strongly recommended | abstract text |
| `Affiliation` | no | affiliation text or list |
| `Year` | no | publication year |
| `precomputed_embedding` | no | precomputed text embedding; skips embedding step when present |

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
| `05_stage_metrics_infer_sources.json` | per-stage runtime and diagnostic metrics |
| `05_go_no_go_infer_sources.json` | run validation summary |

## Runtime notes

- CPU path uses `chars2vec`; ONNX is used automatically when the `cpu_onnx` extra is installed, otherwise falls back to the transformers CPU path.
- The package fails early with a clear error if a requested backend is unavailable or there is insufficient scratch space.

## Further reading

- [Inference workflow](https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/inference_workflow.md)
- [Training workflow](https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/training_workflow.md)
- [Provenance and baseline notes](https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/provenance.md)

## Citation

If you use `ads-and`, cite the software entry in [`CITATION.cff`](CITATION.cff) and the underlying NAND paper:

Vicente Amado Olivo, Wolfgang Kerzendorf, Bangjing Lu, Joshua V. Shields, Andreas Flörs, and Nutan Chen.  
*Practical Author Name Disambiguation under Metadata Constraints: A Contrastive Learning Approach for Astronomy Literature.*  
<https://doi.org/10.1088/1538-3873/ae1e2d>

Related resources:
- NAND repository: <https://github.com/deepthought-initiative/neural_name_dismabiguator>
- LSPO dataset: <https://doi.org/10.5281/zenodo.11489161>
