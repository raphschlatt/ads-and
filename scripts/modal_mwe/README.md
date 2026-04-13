# Modal MWE

Freistehender Roundtrip fuer den ADS-Inferenzpfad:

- lokaler Python-Call
- kleine Parquet-Staging-Dateien aus den echten Inputs
- kompletter Remote-Run auf Modal
- lokale Finalisierung zurueck auf die Original-Parquets

Nicht Teil dieses MWE:

- kein `backend="modal"` im Paket
- kein Web-Endpoint
- kein HTTP
- keine Volumes/Buckets
- keine Kostenlogik

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
- deployed `scripts/modal_mwe/modal_app.py`
- ruft den lokalen Client auf
- schreibt Outputs nach `tmp\modal_mwe_smoke\out`

## Manuell

Deploy:

```powershell
uv run --with modal python -m modal deploy .\scripts\modal_mwe\modal_app.py
```

Run:

```powershell
uv run --with modal python .\scripts\modal_mwe\client_mwe.py `
  --publications path\to\publications.parquet `
  --references path\to\references.parquet `
  --output-dir outputs\modal_mwe_run
```

Live-Logs:

```powershell
uv run --with modal python -m modal app logs ads-and-modal-mwe
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
