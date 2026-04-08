# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. The 2026-04-08 ADS Exact-Graph callback and `finalize()` waves stayed output-identical to the current CPU reference. The 2026-04-08 Arrow fast-path wave then removed the remaining large pair-input bottleneck and reduced `pair_inference.wall_seconds` further to about `192.7s`, with `pair_scoring.arrow_column_extract_seconds` effectively eliminated (`~81.3s -> ~0.03s`). At this point no equally large cold-run code lever is clearly left in the current pipeline.

- Decide whether to stop performance work here and freeze the current cold-run reference, or to spend one final ROI wave on either the SPECTER cold path or export if and only if a realistic `>=20s` gain is identified first.
- Decide whether to install and benchmark optional `numba` bulk-union support in the repo `.venv`; correctness is already preserved without it, but the current validated callback run still used `union_impl = "python"`.
- Keep `chars2vec` GPU as an experiment only until the CPU/GPU block-level drift is understood well enough for a real quality gate.
- Decide whether the historical `srcd52...` LSPO/train baseline should be reconstructed or formally retired as historical-only.
- Decide whether `cuml_gpu` gets a separate supported environment/workflow instead of sharing the standard repo `.venv`.
