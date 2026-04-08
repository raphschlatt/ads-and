# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. The 2026-04-08 ADS Exact-Graph fast-path validation run stayed output-identical to the current CPU reference (`ads_clusters_delta = 0`, `changed_mentions = 0`) and reduced `pair_inference.wall_seconds` from about `549.8s` to `297.3s`. The previous `exact_graph_accumulator_init_seconds ~= 165s` bottleneck is now down to about `6.3s`; the next dominant measured costs are `score_callback_seconds ~= 52.9s` and `arrow_column_extract_seconds ~= 72.6s`.

- Optimize the Exact-Graph callback path next, starting with the `score_callback_seconds` envelope, especially the `union` substep, before touching the remaining `arrow_column_extract_seconds` envelope.
- Decide whether to install and benchmark optional `numba` bulk-union support in the repo `.venv`; correctness is already preserved without it, but the current validated run used `union_impl = "python"`.
- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
