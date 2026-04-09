# ads-and

`ads-and` is a Python package for author name disambiguation in the [SAO/NASA Astrophysics Data System (ADS)](https://ui.adsabs.harvard.edu/). It is designed for one task: take ADS-style parquet inputs and write disambiguated outputs with stable author identifiers.

The bundled baseline model is derived from the [Neural Author Name Disambiguator (NAND)](https://github.com/deepthought-initiative/neural_name_dismabiguator) line of work described in [Practical Author Name Disambiguation under Metadata Constraints: A Contrastive Learning Approach for Astronomy Literature](https://doi.org/10.1088/1538-3873/ae1e2d). NAND was evaluated on the [Large-Scale Physics Open Researcher and Contributor ID (ORCID)-Linked dataset (LSPO)](https://doi.org/10.5281/zenodo.11489161), a large physics and astronomy author name disambiguation dataset built from NASA/ADS records linked to ORCID identities.

The paper reports that NAND achieves up to **94% accuracy** in pairwise disambiguation and **over 95% F1** in clustering on LSPO. The current packaged operational reference in this repository is not byte-identical to the original research stack; the current reproducible LSPO quality reference for the bundled package is **F1 = 0.9702**, **precision = 0.9636**, and **recall = 0.9770** with constraints enabled on the retained five-seed LSPO evaluation workflow.

The default workflow uses the model that ships with the package, so no external model bundle is required.

Package name on PyPI: `ads-and`  
Python import path: `author_name_disambiguation`

## Install

```bash
uv pip install ads-and
```

or

```bash
pip install ads-and
```

Optional CPU ONNX runtime:

```bash
uv pip install "ads-and[cpu_onnx]"
```

`ads-and` requires Python 3.11 or newer.

## Command line

Run disambiguation on one ADS dataset:

```bash
ads-and infer \
  --publications-path data/ads/publications.parquet \
  --references-path data/ads/references.parquet \
  --output-dir outputs/ads_run
```

Add `--json` if you want the final run summary as JSON.

Runtime selection:

- `--runtime auto` prefers a local GPU when CUDA is available and otherwise uses the local CPU path
- `--runtime gpu` forces the local GPU path
- `--runtime cpu` forces the local CPU path

## Python

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

## Input data

`--publications-path` is required. `--references-path` is optional.

Each input record should use ADS-style columns:

| Column | Required | Notes |
| --- | --- | --- |
| `Bibcode` | yes | ADS source identifier |
| `Author` | yes | author list for the record |
| `Year` | no | publication year |
| `Title_en` or `Title` | no | title text |
| `Abstract_en` or `Abstract` | no | abstract text |
| `Affiliation` | no | affiliation text or affiliation list |
| `precomputed_embedding` | no | optional precomputed text embedding |

Records without `Bibcode` or `Author` are skipped.

## Output

The package writes the following files under `output_dir`:

| File | Purpose |
| --- | --- |
| `publications_disambiguated.parquet` | publications with disambiguated author columns |
| `references_disambiguated.parquet` | references with disambiguated author columns when references are provided |
| `source_author_assignments.parquet` | row-level author assignments |
| `author_entities.parquet` | inferred author entities |
| `mention_clusters.parquet` | mention-to-cluster mapping |
| `summary.json` | high-level run summary |
| `05_stage_metrics_infer_sources.json` | detailed runtime and stage metrics |
| `05_go_no_go_infer_sources.json` | run validation summary |

The disambiguated source parquets keep the input columns and add:

- `AuthorUID`
- `AuthorDisplayName`

## Runtime behavior

- The bundled model is embedded in the package.
- The supported product path runs `chars2vec` on CPU.
- CPU inference can use ONNX when the optional `cpu_onnx` extra is installed and usable.
- If ONNX is unavailable or unusable, the package falls back automatically to the local transformers CPU path.
- If a requested run is physically impossible because of missing scratch space or an unsupported explicit backend, the package fails early with a clear error.

## Scope

`ads-and` is an inference package for ADS-style parquet data. It is not presented as a general-purpose author name disambiguation toolkit for arbitrary metadata formats.

The repository also contains training, LSPO evaluation, and baseline-management workflows for research use, but those workflows are not required for normal package use:

- repository: <https://github.com/raphschlatt/Author_Name_Disambiguation>
- inference workflow: <https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/inference_workflow.md>
- training workflow: <https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/training_workflow.md>
- provenance and baseline notes: <https://github.com/raphschlatt/Author_Name_Disambiguation/blob/main/docs/provenance.md>

## Background and citation

If you use `ads-and`, cite the software entry in `CITATION.cff` and the NAND paper:

- original NAND repository: <https://github.com/deepthought-initiative/neural_name_dismabiguator>
- LSPO dataset: <https://doi.org/10.5281/zenodo.11489161>
- paper: *Practical Author Name Disambiguation under Metadata Constraints: A Contrastive Learning Approach for Astronomy Literature*  
  Vicente Amado Olivo, Wolfgang Kerzendorf, Bangjing Lu, Joshua V. Shields, Andreas Flörs, and Nutan Chen  
  <https://doi.org/10.1088/1538-3873/ae1e2d>
