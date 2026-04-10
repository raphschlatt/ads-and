# Inference Workflow

This is a repo-level runbook for ADS inference. Public package users normally
only need the `README.md` usage examples.

## Public Path

The public command uses the bundled Trained NAND Model:

```bash
ads-and infer \
  --publications-path data/ads/publications.parquet \
  --references-path data/ads/references.parquet \
  --output-dir outputs/ads_run \
  --runtime auto
```

Equivalent Python entry point:

```python
from author_name_disambiguation import disambiguate_sources
```

`--references-path` is optional. `--runtime` is `auto`, `gpu`, or `cpu`.

## ADS Full Candidate Run

An ADS Full Candidate Run is the repo-only production-scale inference benchmark
for an inference-only experiment. It does not train or change the Trained NAND
Model.

Run it from a checkout:

```bash
python -m author_name_disambiguation_research run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current
```

Required arguments are `--publications-path`, `--output-root`, and
`--dataset-id`. Useful optional arguments are `--references-path`,
`--model-bundle`, `--scratch-dir`, `--runtime-mode`, `--cluster-backend`,
`--uid-scope`, and `--uid-namespace`.

If `--model-bundle` is omitted, the packaged fixed model bundle is used.

## Runtime Policy

Current product behavior is intentionally conservative:

- `chars2vec` runs on CPU in the product path.
- SPECTER uses GPU when requested and available; CPU mode can use `cpu_auto`,
  which prefers ONNX when available and falls back to transformers.
- `cluster_backend=auto` resolves to `sklearn_cpu`.
- `cuml_gpu` is explicit opt-in only.
- `numba` is optional and is not auto-selected.

The resolved runtime policy and fallbacks are written to run reports before
the expensive stages start.

## Outputs and Reports

The run writes disambiguated sources plus operational reports under
`--output-root`, including:

- `publications_disambiguated.parquet`
- `references_disambiguated.parquet`, when references are provided
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `summary.json`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

## ADS Inference Baseline Comparison

The current ADS inference baseline is `bench_full_v22_fix2`.

Compare an ADS Full Candidate Run against it with:

```bash
python -m author_name_disambiguation_research compare-infer-baseline \
  --baseline-run-id bench_full_v22_fix2 \
  --current-run-id <candidate-run-id> \
  --metrics-root artifacts/exports
```

Keep or promote a candidate only after reviewing runtime, coverage, and output
drift against the ADS inference baseline.
