# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. The 2026-04-07 ADS telemetry validation run showed `pair_inference.unaccounted_wall_seconds ~= 0`, with the dominant measured costs now at `exact_graph_accumulator_init_seconds ~= 165s`, `score_callback_seconds ~= 71s`, and `arrow_column_extract_seconds ~= 100s`.

- Optimize the Exact-Graph path next, starting with `ExactGraphClusterAccumulator(...)` initialization and then the `score_callback_seconds` envelope before touching the smaller remaining `arrow_column_extract_seconds` envelope.
- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
