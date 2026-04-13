# Modal MWE

Freistehender Roundtrip fuer den ADS-Inferenzpfad:

- lokaler Python-Call
- kleine Parquet-Staging-Dateien aus den echten Inputs
- kompletter Remote-Run auf einer ephemeren Modal-App
- lokale Finalisierung zurueck auf die Original-Parquets

Nicht Teil dieses MWE:

- kein `backend="modal"` im Paket
- kein Web-Endpoint
- kein HTTP
- keine Volumes/Buckets
- keine Kostenschaetzung

## Auth

Entweder:

- `uv run --with modal modal setup`
- oder `MODAL_TOKEN_ID` und `MODAL_TOKEN_SECRET` in repo-root `.env`

## Smoke Test

```powershell
.\scripts\modal_mwe\smoke_test.ps1
```

Das Script:

- baut ein kleines Testset unter `tmp\modal_mwe_smoke\input`
- ruft den lokalen Client im `run`-Modus auf
- schreibt Outputs nach `tmp\modal_mwe_smoke\out`
- zeigt den spaeteren Cost-Command an

## Manuell

Run:

```powershell
uv run --with modal python .\scripts\modal_mwe\client_mwe.py `
  run `
  --publications path\to\publications.parquet `
  --references path\to\references.parquet `
  --output-dir outputs\modal_mwe_run
```

Exakte Kosten spaeter abfragen:

```powershell
uv run --with modal python .\scripts\modal_mwe\client_mwe.py `
  cost `
  --output-dir outputs\modal_mwe_run
```

## Inputs / Outputs

Transportiert werden nur die ADS-Spalten, die der Inferenzpfad braucht:

- `Bibcode` / `bibcode`
- `Author` / `author`
- `Title_en` / `Title` / `title`
- `Abstract_en` / `Abstract` / `abstract`
- `Year` / `year`
- `Affiliation` / `Affilliation` / `aff`

Finale Outputs bleiben voll, weil die Rueckfuehrung lokal auf die Original-Parquets passiert.

`summary.json` enthaelt nur die Modal-Lookup-Daten fuer spaetere Ist-Kosten:

- `app_id`
- `run_started_at_utc`
- `run_finished_at_utc`
- `exact_cost_available_after_utc`

Wenn der Workspace programmatic billing reports unterstuetzt, schreibt der `cost`-Command spaeter ein `modal_cost_report.json` mit den exakten Ist-Kosten.
