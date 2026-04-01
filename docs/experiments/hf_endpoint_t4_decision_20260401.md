# HF Endpoint Decision 2026-04-01

This repo now keeps exactly one public HF path:

- dedicated Hugging Face Inference Endpoint
- `AWS / eu-west-1 / Nvidia T4 / x1`
- `allenai/specter`
- client-side truncation to `256`
- client batching `chunk_size=64`
- client parallelism `concurrency=8`

Why this one stayed:

- it is standard HF serving, not custom infrastructure
- it completed real ADS-like runs
- it was the best standard remote HF path we measured

Rejected paths:

- router `httpx` path
  - timed out or stalled on real larger slices
  - too much transport/profile complexity for too little reliability
- router `requests` batching
  - worked, but was too slow on real data
- dedicated `A100` endpoint
  - no meaningful win over the kept T4 standard path
  - more cost for no clear package benefit
- TEI / custom image path
  - blocked by missing `tokenizer.json` in `allenai/specter`
  - too custom for the package direction anyway

This note is intentionally the only permanent memory of the discarded HF branches. The corresponding code paths were deleted to keep the package lean.
