# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total.

- Validate the second-wave pair telemetry on a fresh ADS run and then choose the next optimization target from the measured dominant component: `exact_graph_accumulator_init_seconds`, `score_callback_seconds`, or the remaining `arrow_column_extract_seconds` envelope.
- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
