# ToDo

Note: `runtime.pair_inference.wall_seconds` is now the authoritative pair-stage total. The 2026-04-08 ADS optimization session kept the current CPU reference output stable while reducing cold-run time materially, culminating in the Arrow fast-path state plus the kept export frame-reuse improvement. The 2026-04-08 hardware-adaptive runtime hardening wave then fixed the product stance and default runtime policy around that faster path.

Closed decisions now reflected in docs/code:

- `chars2vec` GPU is out as a product path for `infer_sources`; product inference uses CPU-only `chars2vec`
- `numba` remains optional and is not prioritized or auto-selected
- `cuml_gpu` remains optional/special and is not the standard `auto` path
- the historical ADS baseline manifest stays in place; the faster 2026-04-08 package state is documented operationally instead of being promoted here as a new historical baseline

Remaining work is now mostly product/ops, not performance chasing:

- Run and retain one explicit real CPU-only acceptance smoke for the hardware-adaptive auto policy, so the docs are backed by a non-mocked end-to-end artifact as well.
- Decide whether the historical `srcd52...` LSPO/train baseline should be formally retired as historical-only or reconstructed on purpose.
- If `cuml_gpu` is ever supported beyond experimental use, define it as a separate documented environment/workflow instead of broadening the standard repo `.venv`.
