# Data Contracts

## Public Source-Infer Contract

`run-infer-sources` und `run_infer_sources()` erwarten Source-Records, keine Mentions.

Mindestfelder pro Input-Record:

- `Bibcode`
- `Author`

Optionale Felder:

- `Year`
- `Title_en` oder `Title`
- `Abstract_en` oder `Abstract`
- `Affiliation`
- legacy alias: `embedding`
- canonical field: `precomputed_embedding`

Records ohne `Bibcode` oder `Author` werden intern nicht in Mentions umgewandelt. Source-mirrored Outputs behalten rohe Zeilen mit leerem `Author`-Feld trotzdem bei und schreiben dafuer leere `AuthorUID`-/`AuthorDisplayName`-Listen.

### Aktiver Text-Embedding-Vertrag

Fuer das aktuelle produktive Bundle bedeutet `precomputed_embedding` nicht “beliebiger 768-dim Vektor”, sondern:

- Modellfamilie: `allenai/specter`
- Dimension: `768`
- Textaufbau: `Title [SEP] Abstract`
- Pooling: `cls_first_token`
- Tokenisierung: `truncation=True`, `max_length=256`

Nur gleiche Dimension reicht nicht als Bundle-Kompatibilitaet.

### CPU-First Precompute

Die offizielle Welle-1-Story fuer CPU-User ist ein separater Precompute-Schritt:

```bash
export HF_TOKEN=...
author-name-disambiguation precompute-source-embeddings \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/precomputed/ads_prod_current
```

Die angereicherten Dateien werden immer als Parquet unter `output_root` geschrieben:

- `publications_precomputed.parquet`
- optional `references_precomputed.parquet`
- `precompute_source_embeddings_report.json`

## Public Infer Outputs

Ein erfolgreicher Lauf unter `output_root` schreibt:

- `publications_disambiguated.{parquet|jsonl}`
- optional `references_disambiguated.{parquet|jsonl}`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

### Source-mirrored Outputs

Die disambiguierten Source-Dateien behalten alle Inputspalten und ergänzen:

- `AuthorUID: list[str]`
- `AuthorDisplayName: list[str]`

Beide Listen sind positionsparallel zu `Author`.
Bei rohen Source-Zeilen ohne Autoren sind beide Listen leer.

### `source_author_assignments.parquet`

Spalten:

- `source_type`
- `source_row_idx`
- `bibcode`
- `author_idx`
- `author_raw`
- `author_uid`
- `author_uid_local`
- `author_display_name`
- `assignment_kind`
- `canonical_mention_id`

`assignment_kind`:

- `canonical`
- `projected_duplicate`
- `fallback_unmatched`

### `author_entities.parquet`

Spalten:

- `author_uid`
- `author_uid_local`
- `author_display_name`
- `aliases`
- `mention_count`
- `document_count`
- `unique_mention_count`
- `display_name_method`

`display_name_method` ist aktuell immer `most_frequent_alias`.

## Internal Core Tables

Diese Tabellen bleiben interne Arbeitsrepräsentationen:

`mentions.parquet`:

- `mention_id`
- `bibcode`
- `author_idx`
- `author_raw`
- `title`
- `abstract`
- `year`
- `source_type`
- `block_key`
- optional `aff`

`pairs_*.parquet`:

- `pair_id`
- `mention_id_1`
- `mention_id_2`
- `block_key`
- `split`
- `label`

`pair_scores.parquet`:

- `pair_id`
- `mention_id_1`
- `mention_id_2`
- `block_key`
- `cosine_sim`
- `distance`

`clusters.parquet`:

- `mention_id`
- `block_key`
- `author_uid`
- optional `author_uid_local`

## Train Artifacts

`run-train-stage` schreibt weiterhin:

- `03_train_manifest.json`
- `04_clustering_config_used.json`
- `05_stage_metrics_<stage>.json`
- `05_go_no_go_<stage>.json`
- `06_clustering_test_report.{json,md}`
- `06_clustering_test_summary.csv`
- `06_clustering_test_per_seed.csv`

## Model Bundle Contract

Bundle-Inhalt:

- `bundle_manifest.json`
- `checkpoint.pt`
- `model_config.yaml`
- `clustering_resolved.json`

`bundle_manifest.json` enthält mindestens:

- `bundle_schema_version`
- `source_model_run_id`
- `checkpoint_hash`
- `selected_eps`
- `best_threshold`
- `precision_mode`
- `embedding_contract`

`run-infer-sources --model-bundle <dir>` konsumiert dieses Bundle direkt.

`embedding_contract` dokumentiert den erwarteten Text- und Name-Embedding-Raum des Bundles. Bestehende `v1`-Bundles ohne dieses Feld bleiben lesbar; der Vertrag wird dann aus `model_config.yaml` abgeleitet.
