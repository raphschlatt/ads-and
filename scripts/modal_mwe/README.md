# Modal Smoke Helper

Smoke-Test-Harness fuer das in-Package-Modal-Backend. Baut mit `build_testset.py`
Mini-Parquet-Inputs und ruft dann direkt `ads-and infer --backend modal` auf.

## Auth

Entweder:

- `uv run --extra modal modal setup`
- oder `MODAL_TOKEN_ID` und `MODAL_TOKEN_SECRET` in repo-root `.env`

## Smoke Test

```powershell
.\scripts\modal_mwe\smoke_test.ps1
```

Das Script baut ein kleines Testset und ruft dann den echten Paketpfad:

```powershell
uv run --extra modal ads-and infer --backend modal ...
uv run --extra modal ads-and cost --output-dir ...
```
