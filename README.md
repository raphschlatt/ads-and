# ads-and

`ads-and` is a focused Python package for ADS author name disambiguation.

The public product story is intentionally simple:

- install one package
- provide curated ADS parquet inputs
- call one CLI command or one Python function
- use the embedded ADS baseline model automatically
- let the package choose a safe local CPU/GPU runtime
- receive a disambiguated ADS dataset plus compact run artifacts

The Python import path remains `author_name_disambiguation`.

## Install

```bash
uv pip install ads-and
```

Optional CPU ONNX runtime:

```bash
uv pip install "ads-and[cpu_onnx]"
```

## CLI

Disambiguate one ADS dataset with the bundled baseline model:

```bash
ads-and infer \
  --publications-path data/ads/publications.parquet \
  --references-path data/ads/references.parquet \
  --output-dir outputs/ads_run
```

JSON output:

```bash
ads-and infer \
  --publications-path data/ads/publications.parquet \
  --references-path data/ads/references.parquet \
  --output-dir outputs/ads_run \
  --json
```

Runtime selection:

- `--runtime auto` chooses local GPU when CUDA is available, otherwise local CPU
- `--runtime gpu` forces the local GPU path
- `--runtime cpu` forces the local CPU path

The public package does not require an external model bundle for the default workflow.

## Python API

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

## Input Contract

Input fields per source record:

- required: `Bibcode`
- required: `Author`
- optional: `Year`
- optional: `Title_en` or `Title`
- optional: `Abstract_en` or `Abstract`
- optional: `Affiliation`
- optional legacy alias: `embedding`
- optional canonical field: `precomputed_embedding`

Records without `Bibcode` or `Author` are skipped during inference.

The bundled ADS model expects the current production text-embedding contract:

- model family: `allenai/specter`
- dimension: `768`
- text assembly: `Title [SEP] Abstract`
- pooling: first-token / CLS from `last_hidden_state[:, 0, :]`
- tokenization: truncation with `max_length=256`

## Outputs

Inference writes under `output_dir`:

- `publications_disambiguated.parquet`
- optional `references_disambiguated.parquet`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `summary.json`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

The disambiguated source files keep the input columns and add:

- `AuthorUID`
- `AuthorDisplayName`

## Runtime Notes

- The bundled model is embedded in the package.
- `chars2vec` is CPU-only in the supported product path.
- `cluster_backend=auto` resolves to the standard CPU clustering path.
- `cpu_auto` prefers ONNX on CPU when the optional ONNX extra is installed and usable; otherwise it falls back to the local transformers CPU path.
- If a requested run is physically impossible because of missing scratch space or an unsupported explicit backend, the package fails early with a clear error.

## Repo-Only Research Workspace

Training, LSPO quality checks, baselines, benchmarks, cleanup, and experimental runtime paths remain available from a repository checkout, but they are not part of the public PyPI contract.

Editable repo install:

```bash
uv pip install -e ".[dev,research]"
```

Repo-only research CLI:

```bash
python -m author_name_disambiguation_research -h
```

Repo-only reference docs:

- `docs/inference_workflow.md`
- `docs/training_workflow.md`
- `docs/provenance.md`
