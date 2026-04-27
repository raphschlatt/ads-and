# Final Clustering Test Report

- model_run_id: `full_20260218T111506Z_cli02681429`
- run_stage: `full`
- generated_utc: `2026-04-27T12:47:31.352724+00:00`
- wall_seconds: `3362.790467`
- selected_eps: `0.350000`
- min_samples: `1`
- metric: `precomputed`
- seeds_expected: `[1, 2, 3, 4, 5]`
- seeds_evaluated: `[1, 2, 3, 4, 5]`

## Summary

| variant | accuracy_mean | accuracy_sem | precision_mean | precision_sem | recall_mean | recall_sem | f1_mean | f1_sem | n_pairs_mean | n_pairs_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dbscan_no_constraints | 0.931701 | 0.002224 | 0.950351 | 0.000540 | 0.978099 | 0.002399 | 0.964019 | 0.001208 | 1679704.000000 | 8398520 |
| dbscan_with_constraints | 0.943947 | 0.002494 | 0.963580 | 0.000427 | 0.977015 | 0.002492 | 0.970245 | 0.001353 | 1679704.000000 | 8398520 |

## Delta (with_constraints - no_constraints)

| accuracy | precision | recall | f1 |
|---:|---:|---:|---:|
| 0.012246 | 0.013229 | -0.001084 | 0.006226 |

## Per Seed

| seed | variant | threshold | accuracy | precision | recall | f1 | n_pairs | checkpoint |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | dbscan_no_constraints | 0.483000 | 0.928424 | 0.950178 | 0.974598 | 0.962233 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed1.pt |
| 2 | dbscan_no_constraints | 0.516000 | 0.928660 | 0.948636 | 0.976628 | 0.962429 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed2.pt |
| 3 | dbscan_no_constraints | 0.484000 | 0.931602 | 0.952032 | 0.976073 | 0.963903 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed3.pt |
| 4 | dbscan_no_constraints | 0.516000 | 0.940311 | 0.950531 | 0.987600 | 0.968711 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed4.pt |
| 5 | dbscan_no_constraints | 0.502000 | 0.929507 | 0.950376 | 0.975595 | 0.962820 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed5.pt |
| 1 | dbscan_with_constraints | 0.483000 | 0.940195 | 0.963139 | 0.973328 | 0.968207 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed1.pt |
| 2 | dbscan_with_constraints | 0.516000 | 0.940841 | 0.962147 | 0.975132 | 0.968596 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed2.pt |
| 3 | dbscan_with_constraints | 0.484000 | 0.942881 | 0.964287 | 0.975061 | 0.969644 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed3.pt |
| 4 | dbscan_with_constraints | 0.516000 | 0.953748 | 0.964491 | 0.986897 | 0.975566 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed4.pt |
| 5 | dbscan_with_constraints | 0.502000 | 0.942071 | 0.963835 | 0.974654 | 0.969214 | 1679704 | artifacts/checkpoints/full_20260218T111506Z_cli02681429/full_20260218T111506Z_cli02681429_seed5.pt |
