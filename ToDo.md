# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total.

- Instrument the remaining hidden pair-stage time. The validation run `ads_full_pair_timing_reconciled_20260407_v1` still shows about `154.8s` `runtime.pair_inference.unaccounted_wall_seconds` and about `65.9s` hidden inside `runtime.pair_scoring.wall_seconds`.
- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
