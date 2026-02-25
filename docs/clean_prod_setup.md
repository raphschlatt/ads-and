# Clean Production Setup (ADS)

## Behalten (für funktionalen Betrieb + Qualitätsvergleich)

- Full-Modell (produktiv):
  - `artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1`
  - `artifacts/checkpoints/full_20260218T111506Z_cli02681429`
  - `artifacts/metrics/full_20260218T111506Z_cli02681429`
- Referenzrun (Vergleich):
  - `artifacts/checkpoints/mid_20260217T173829Z_cli60965158`
  - `artifacts/metrics/mid_20260217T173829Z_cli60965158`
  - `artifacts/models/mid_20260217T173829Z_cli60965158`
  - `artifacts/embeddings/mid_20260217T173829Z_cli60965158`

## ADS-Datenablage (neu)

Lege dein echtes ADS-Set hier ab:

- `data/raw/ads/ads_prod_current/publications.jsonl`
- optional `data/raw/ads/ads_prod_current/references.jsonl`

Konfiguration ist darauf gesetzt in `configs/paths.local.yaml`.

## Inferenz starten

Mit Full-Run-ID:

```bash
python -m src.cli run-infer-ads \
  --dataset-id ads_prod_current \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --infer-stage full \
  --cpu-sharding auto \
  --cpu-workers auto \
  --cluster-backend auto
```

Oder mit Bundle:

```bash
python -m src.cli run-infer-ads \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1 \
  --infer-stage full \
  --cpu-sharding auto \
  --cpu-workers auto \
  --cluster-backend auto
```

## Cleanup-Restore

Alle entfernten Entwicklungsdaten wurden verschoben nach:

- `/home/ubuntu/trash/nand_cleanup_20260223T000000Z_prod_clean/moved`

Du kannst einzelne Ordner bei Bedarf zurückverschieben.
