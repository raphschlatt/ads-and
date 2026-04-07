# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. Treat `unaccounted_wall_seconds` as the follow-up signal if the breakdown drifts again.

- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
