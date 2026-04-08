# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. The 2026-04-08 ADS Exact-Graph callback run stayed output-identical to the current CPU reference and reduced `pair_inference.wall_seconds` to about `281.6s`. The follow-up 2026-04-08 `finalize()` wave also stayed output-identical and brought `pair_inference.wall_seconds` down again to about `278.8s`, `clustering.wall_seconds` down to about `22.8s`, and `export.wall_seconds` down to about `61.1s`. The new clustering breakdown now shows `finalize_total_seconds ~= 22.3s`, of which only `connected_components_seconds_total ~= 6.9s`, `finalize_label_materialization_seconds ~= 2.8s`, and `finalize_output_frame_seconds ~= 0.04s` are explicitly accounted; the remaining hidden finalize envelope is therefore still the next CPU-side target, ahead of `arrow_column_extract_seconds ~= 81.3s`.

- Instrument the remaining hidden work inside `ExactGraphClusterAccumulator.finalize()` next, especially edge chunk concatenation/deduplication and any still-unmeasured per-block bookkeeping between component solve and label output, before touching the remaining `arrow_column_extract_seconds` envelope.
- Decide whether to install and benchmark optional `numba` bulk-union support in the repo `.venv`; correctness is already preserved without it, but the current validated callback run still used `union_impl = "python"`.
- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
