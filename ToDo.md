# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. The 2026-04-08 ADS optimization session kept the current CPU reference output stable while reducing cold-run time materially, culminating in the Arrow fast-path state plus the kept export frame-reuse improvement. The 2026-04-08 hardware-adaptive runtime hardening wave then fixed the product stance and default runtime policy around that faster path. The current release-engineering wave splits the project into a public inference package plus a repo-only research workspace.

Closed decisions now reflected in docs/code:

- `chars2vec` GPU is out as a product path for `infer_sources`; product inference uses CPU-only `chars2vec`
- `numba` remains optional and is not prioritized or auto-selected
- `cuml_gpu` remains optional/special and is not the standard `auto` path
- the historical ADS baseline manifest stays in place; the faster 2026-04-08 package state is documented operationally instead of being promoted here as a new historical baseline
- the public PyPI surface is now inference-only and centered on the bundled ADS baseline model
- training, LSPO quality, baselines, and experimental workflows remain repo-only and are no longer part of the public package contract

Remaining work is now mostly product/ops, not performance chasing:

- Decide whether the historical `srcd52...` LSPO/train baseline should be formally retired as historical-only or reconstructed on purpose.
- If CPU-only speed becomes a first-class product target, benchmark ONNX thread tuning or a host-aware CPU selector before making a stronger ONNX speed claim; the 2026-04-08 repo-host smoke showed ONNX CPU was functional but slightly slower than the transformers CPU path there.
- If `cuml_gpu` is ever supported beyond experimental use, define it as a separate documented environment/workflow instead of broadening the standard repo `.venv`.
- tighten the public release gate around wheel/sdist install smoke, bundled-model resolution, and one-command ADS inference from a fresh environment
