# chars2vec Wave `perf_pkg2_chars_v1`

Date:
- `2026-03-19`

Status:
- `LSPO Gate Run` failed
- `ADS Full Candidate Run` was not started
- experiment is documented but not promoted

## Scope

Experiment type:
- inference-only experiment

Changed object:
- chars2vec inference execution path in [embed_chars2vec.py](/home/ubuntu/Author_Name_Disambiguation/src/author_name_disambiguation/features/embed_chars2vec.py)

What stayed fixed:
- `Raw LSPO Source`
- `Trained NAND Model`
- `trained model baseline`: `full_20260218T111506Z_cli02681429`
- `ADS inference baseline`: `bench_full_v22_fix2`

## What Was Tested

Primary idea:
- keep `eng_50`
- change chars2vec from historical `predict(batch_size=32)` behavior to auto-batching on GPU

Secondary lab idea:
- add `direct_call` mode for benchmark-only exploration

## Evidence

Microbenchmark artifact:
- [chars2vec_modes_ads_sample200k_perf_pkg2_chars_v1.json](/home/ubuntu/Author_Name_Disambiguation/artifacts/benchmarks/chars2vec_modes_ads_sample200k_perf_pkg2_chars_v1.json)

Microbenchmark result on ADS sample:
- `200000` names
- `110503` unique names
- `predict+32`: `91.9586s` wall
- `predict+auto`: `12.0438s` wall
- `direct_call`: `12.1896s` wall

LSPO gate artifacts:
- [06_clustering_test_report__perf_pkg2_chars_v1.json](/home/ubuntu/Author_Name_Disambiguation/artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__perf_pkg2_chars_v1.json)
- [99_compare_cluster_report_to_baseline__perf_pkg2_chars_v1.json](/home/ubuntu/Author_Name_Disambiguation/artifacts/metrics/full_20260218T111506Z_cli02681429/99_compare_cluster_report_to_baseline__perf_pkg2_chars_v1.json)

LSPO gate result:
- `decision = rollback_to_baseline`
- `delta_f1_mean = -0.000005876`
- `delta_precision_mean = -0.000011883`
- `delta_accuracy_mean = -0.000011431`
- `delta_recall_mean = +0.000000255`

Additional local diagnostics after the gate:
- `predict+32` vs `predict+auto` on `50000` names was not all-close even at `atol=1e-5`
- measured `max_abs_diff = 0.0016091568`
- measured `mean_abs_diff = 0.0000755116`
- `direct_call` also drifted from `predict+32` and emitted a Keras input-structure warning
- a batch sweep on `10000` names showed that every tested batch size other than `32` (`64`, `128`, `256`, `512`) drifted from the historical `32` path

## Interpretation

What the result means:
- the speed-up is real
- the speed-up is not a free runtime-only win in this repo
- changing chars2vec batch behavior changes embeddings numerically
- those small embedding deltas are enough to move LSPO gate metrics

Why this matters:
- in this project, `chars2vec` sits upstream of pair scoring and clustering
- tiny numeric changes can move borderline cluster decisions
- under the current gate policy, this experiment is therefore a no-go

Relation to upstream chars2vec:
- upstream `vectorize_words(...)` uses the model-local cache and calls `embedding_model.predict([embeddings_pad_seq])`
- upstream does not document a GPU-first or large-batch equivalence guarantee
- for this repo, the historical reference path should be treated as `predict` with effective batch size `32`

## Decision

Do not promote from this wave:
- auto-batching as the default chars2vec path
- `direct_call` as production behavior

Do preserve from this wave:
- the benchmark artifact
- the benchmark helper script
- the explicit lesson that chars2vec batch-size changes are quality-relevant here

## Recommended Next Step

Immediate repo-hygiene recommendation:
- restore the default inference behavior to the historical chars2vec path before running further promoted inference experiments

Recommended next chars2vec wave:
- optimize only paths that preserve the historical numerical behavior as closely as possible

Good candidates for the next wave:
- exact-path profiling of chars2vec preprocessing and padding overhead while keeping effective batch size `32`
- exact-path cache and lifecycle improvements that do not change the numerical execution path
- only after that, re-run `LSPO Gate Run`

Not recommended as the next move:
- another `ADS Full Candidate Run` from `perf_pkg2_chars_v1`
- further tuning of large-batch chars2vec without first deciding whether small output drift is acceptable policy
