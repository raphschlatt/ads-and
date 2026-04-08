# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. The 2026-04-08 ADS Exact-Graph callback validation run stayed output-identical to the current CPU reference (`ads_clusters_delta = 0`, `changed_mentions = 0`) and reduced `pair_inference.wall_seconds` from about `297.3s` to `281.6s`. Inside pair scoring, `score_callback_seconds` fell from about `52.9s` to `16.0s`, `score_callback_union_seconds` dropped to `0.0s`, and finalize-side `connected_components_seconds_total` now measures only about `7.7s`. The next dominant measured costs are `arrow_column_extract_seconds ~= 80.5s` and the still-unaccounted remainder inside `clustering.wall_seconds ~= 23.4s`.

- Instrument and optimize `ExactGraphClusterAccumulator.finalize()` next, especially label/materialization work outside `connected_components_seconds_total`, before touching the remaining `arrow_column_extract_seconds` envelope.
- Decide whether to install and benchmark optional `numba` bulk-union support in the repo `.venv`; correctness is already preserved without it, but the current validated callback run still used `union_impl = "python"`.
- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
