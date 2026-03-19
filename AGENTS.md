# AGENTS.md

## Purpose

This file defines the required vocabulary for this repo so that humans and agents do not talk past each other.

Use these terms exactly.

## The Four Core Objects

### 1. Raw LSPO Source

Meaning:
- The immutable LSPO source data.

Canonical path in this repo:
- `data/raw/lspo/LSPO_v1.parquet`

Rules:
- This is the raw gold source.
- Do not call derived parquet caches "the raw LSPO data".
- Do not modify this file during experiments.

### 2. Trained NAND Model

Meaning:
- The already trained pair-scoring model and its checkpoints/bundle.
- This is the model that was trained from LSPO and is then reused for evaluation and ADS inference.

Canonical artifact examples in this repo:
- training run id: `full_20260218T111506Z_cli02681429`
- bundle: `artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1`

Rules:
- For inference-only experiments, this model stays fixed.
- Do not say "we train LSPO" when you mean "we run LSPO evaluation".
- `run-train-stage` is only for a real model-training experiment that intentionally changes weights, training config, or training-time representation semantics.

### 3. LSPO Gate Run

Meaning:
- The first gate for an inference-only experiment.
- This is an LSPO evaluation run using the existing trained NAND model.
- It checks whether a code change in inference behavior causes unacceptable deviation on the LSPO benchmark.

Canonical command:
- `author-name-disambiguation run-cluster-test-report`

What it does:
- Rebuilds the LSPO evaluation subset from the Raw LSPO Source.
- Reuses an existing trained model run via `--model-run-id`.
- Recomputes embeddings / pair scores / clustering for LSPO evaluation.
- Writes `06_clustering_test_report*.json`.

What it does not do:
- It does not retrain the NAND model.
- It does not change model weights.
- It is not an ADS full run.

Use this term:
- "LSPO gate run"

Do not use these terms for this step:
- "training candidate"
- "LSPO train candidate"
- "baseline training rerun"

### 4. ADS Full Candidate Run

Meaning:
- The second gate for an inference-only experiment.
- This is the real ADS production-scale inference benchmark using the existing exported model bundle.

Canonical command:
- `author-name-disambiguation run-infer-sources`

Current ADS inference baseline in this repo:
- `bench_full_v22_fix2`

What it does:
- Runs full ADS inference with the current model bundle and current inference code.
- Measures runtime, coverage, and output drift.
- Is compared against the ADS inference baseline.

What it does not do:
- It does not retrain the NAND model.

Use this term:
- "ADS full candidate run"

## There Are Two Different Baselines

This repo has two different baseline concepts. Always name which one you mean.

### A. Trained Model Baseline

Meaning:
- The fixed trained NAND model run used for LSPO gate evaluation and for exported bundles.

Example:
- `full_20260218T111506Z_cli02681429`

Use this phrase:
- "trained model baseline"

### B. ADS Inference Baseline

Meaning:
- The current promoted full ADS inference result used for runtime/output comparison.

Example:
- `bench_full_v22_fix2`

Use this phrase:
- "ADS inference baseline"

Never just say:
- "baseline"

Always say:
- "trained model baseline"
- or "ADS inference baseline"

## Experiment Types

### Inference-Only Experiment

Examples:
- clustering changes
- chars2vec execution-path changes
- SPECTER runtime changes
- pair-scoring runtime changes

Correct workflow:
1. Keep the Trained NAND Model fixed.
2. Run an LSPO Gate Run with that fixed model.
3. If LSPO gate is acceptable, run an ADS Full Candidate Run.
4. Compare against the ADS inference baseline.
5. Keep the code change only if the ADS full candidate is worth keeping.

Important:
- For an inference-only experiment, do not run `run-train-stage`.

### Model-Training Experiment

Examples:
- changing model architecture
- changing training features in a way that requires relearning weights
- changing training config / seeds / threshold selection policy

Correct workflow:
1. Run `run-train-stage`.
2. Run `run-cluster-test-report` for that newly trained run.
3. Optionally export a new bundle.
4. Only after that, run ADS inference with the new bundle if promotion is justified.

Important:
- This is the only case where "training a candidate" is the right phrase.

## Canonical Names To Use In Conversation

Use these names exactly:

- "Raw LSPO Source"
- "Trained NAND Model"
- "LSPO Gate Run"
- "ADS Full Candidate Run"
- "trained model baseline"
- "ADS inference baseline"
- "inference-only experiment"
- "model-training experiment"

Avoid these ambiguous names:

- "LSPO train candidate"
- "baseline run" without qualifier
- "the model" without saying whether you mean trained weights or inference code path
- "LSPO testset" when you really mean the LSPO Gate Run

## What The Current chars2vec Experiment Is

Current experiment tag:
- `perf_pkg2_chars_v1`

Experiment type:
- inference-only experiment

What is being changed:
- the chars2vec inference execution path and batch behavior

What must stay fixed:
- the Trained NAND Model
- the Raw LSPO Source

Therefore the correct next step is:
- run an LSPO Gate Run with the existing trained model baseline

The correct first command is conceptually:
- `author-name-disambiguation run-cluster-test-report --model-run-id <trained model baseline> ... --report-tag perf_pkg2_chars_v1`

The wrong command for this experiment is:
- `author-name-disambiguation run-train-stage ...`

## Aborted Run Hygiene

If an agent starts the wrong type of run:
- stop it immediately
- delete only that candidate run's own artifacts
- do not touch the trained model baseline
- do not touch the ADS inference baseline
- do not touch shared caches unless they belong only to that mistaken candidate run

## One-Sentence Mental Model

For inference-only experiments, we keep the Trained NAND Model fixed, use an LSPO Gate Run as Gate 1, then use an ADS Full Candidate Run as Gate 2, and only keep the code change if Gate 1 is acceptable and Gate 2 beats the ADS inference baseline.

## Experiment Memory

Permanent experiment notes live here:
- `docs/experiments/`

Current note for the active chars2vec wave:
- `docs/experiments/perf_pkg2_chars_v1.md`

Rule:
- Keep vocabulary and workflow rules in `AGENTS.md`.
- Keep concrete experiment outcomes, measured numbers, and next-step recommendations in `docs/experiments/`.

## chars2vec Guardrails For This Repo

Upstream chars2vec behavior to treat as the historical reference path:
- lowercase words
- deduplicate with `np.unique`
- reuse the model-local `cache`
- call `embedding_model.predict([embeddings_pad_seq])`

Implication for this repo:
- A chars2vec batch-size change is not automatically a "runtime-only" change.
- In this repo, changing the effective chars2vec batch size can change embeddings numerically and can therefore move LSPO Gate Run metrics.

Current project lesson from `perf_pkg2_chars_v1`:
- The large-batch chars2vec path was much faster in microbenchmarks.
- It did not pass the LSPO Gate Run under the current no-drift policy.
- Therefore large-batch chars2vec must not be treated as promoted behavior unless we explicitly change policy or prove equivalence.

## LSPO Control-Run Reproducibility Rule

When using an `LSPO Gate Run` as a control run against a historical report:
- check `subset_cache_key_computed`
- check `subset_cache_key_expected`
- check `subset_verification_mode`

Interpretation rule:
- if `subset_cache_key_computed` differs from the historical expected key, the run is not a strict historical reproduction
- if `subset_verification_mode = legacy_compat`, treat the run as a compatibility sanity-check, not as a proof of exact historical equivalence

Current known example:
- historical baseline report: `subset_cache_key_computed = full_seed11_targetfull_cfg0dbcdaf9_srcd52b159f766e`
- newer compatibility-path control runs: `subset_cache_key_computed = full_seed11_targetfull_cfg0dbcdaf9_srcb2c9203fe342`

Implication:
- do not blame the current inference code path alone for a control-run drift until the LSPO subset key matches the historical baseline
