# infer_sources speedup wave

## Current baseline

- Produktiver ADS-`chars2vec`-Pfad ist wieder auf `execution_mode="predict"` plus `batch_size=auto` gesetzt.
- `LSPO` bleibt bewusst auf `predict(32)` als separater Qualitäts- und Benchmarkpfad.
- Historische chars2vec-Entscheidung und Messwerte stehen in `docs/experiments/perf_pkg2_chars_v1.md`.
- `Gate zuerst` bleibt die Regel: schnellere Varianten werden erst nach bestandenem LSPO- und ADS-Gate promoted.

## Current regressions / fixes in dieser Welle

- `chars2vec` nutzt wieder dieselbe Live-Progress-Mechanik wie die anderen Stages:
  - TensorFlow-`stderr` wird nur noch in der Bootstrap-Phase gefiltert
  - `predict()` selbst laeuft wieder mit echtem TTY-`stderr`, damit Balken sichtbar bleiben
- `chars2vec` schreibt jetzt eine strukturierte `tensorflow_runtime`-Diagnose in die Runtime-Meta.
- `infer_sources` warnt jetzt sichtbar, wenn PyTorch CUDA sieht, TensorFlow fuer `chars2vec` aber nur auf CPU faellt.
- Neues Repo-Doctor-Script: `scripts/ops/gpu_env_doctor.py`
  - JSON-Ausgabe
  - Human-Readable-Ausgabe
  - `--require-tensorflow-gpu`
- Repo-GPU-Standard ist jetzt `/home/ubuntu/Author_Name_Disambiguation/.venv`, nicht mehr das alte Shared-Host-`.venv`.
- Zielstack ist wieder klar `cu126/cu12`, nicht `cu13`.
- Exact-Graph-Clustering sendet wieder blockweisen Fortschritt statt nur eines finalen `100%`-Signals.
- Balanced Progress bleibt bewusst begrenzt:
  - sichtbar: `chars2vec`, `pair_building`, `exact-graph clustering`
  - weiter stumm: nested `Encode mentions` und `Score batches`
- Pair-Runtime-Meta ist bereinigt:
  - `pair_building.wall_seconds` und `pair_scoring.wall_seconds` sind wieder getrennt
  - `pair_building.sort_parquet_seconds` weist den globalen Sort-Schritt separat aus
- Exact-Graph-Meta ist entwirrt:
  - `mapping_seconds_total`
  - `constraint_apply_seconds_total`
  - `connected_components_seconds_total`
  - `dbscan_seconds_total=0.0` im Exact-Graph-Pfad, damit die Metrik nicht weiter falsch benannt ist
- SPECTER schreibt die effektiv sichtbare `TOKENIZERS_PARALLELISM`-Einstellung in die Runtime-Meta.
- Intermediate `pairs.parquet` enthaelt jetzt numerische Helper-Spalten fuer den Hot-Path:
  - `mention_idx_1`
  - `mention_idx_2`
  - `block_idx`
- NAND-Scoring und Exact-Graph nutzen diese Helper-Spalten bevorzugt und fallen fuer Legacy-Artefakte sauber auf String-Mapping zurueck.

## Next gated experiments

1. Repo-`.venv` mit `gpu_env_doctor.py` auf gesunden `cu126/cu12`-Status bringen.
2. `chars2vec exact32` CPU vs GPU auf der aktuellen Maschine erneut benchmarken.
3. Produktives ADS `chars2vec auto` auf repariertem GPU-Stack erneut benchmarken.
2. SPECTER A/B mit unveraenderter vs nicht erzwungener `TOKENIZERS_PARALLELISM`-Einstellung laufen lassen.
4. ADS Full Run nach dieser Welle neu messen und gegen die Baseline vergleichen.
5. Nur bei ausreichendem Netto-Gewinn den naechsten groesseren Refactor anfassen:
   - Pair-Building auf block spans statt Block-DataFrames
   - Exact-Graph-Union-Find in kompilierten Pfad schieben

## Repo hygiene

- `Infer_MWE.ipynb` bleibt das minimale Package-MWE.
- `Infer_Integration_MWE.ipynb` bleibt das Integrations-MWE fuer externen `progress_handler`.
- `Test.ipynb` ist ein alter Scratch-/Lab-Workspace und kein produktives MWE mehr; er soll entfernt bleiben.
