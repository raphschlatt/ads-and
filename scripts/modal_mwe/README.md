# Modal Smoke Helper

Die echte Modal-Logik sitzt jetzt im Paket.

Dieses Verzeichnis bleibt nur fuer:

- ein kleines Testset
- einen Smoke-Test in `pwsh`
- einen duennen Kompatibilitaets-Wrapper fuer alte `run`/`cost`-Aufrufe

## Auth

Entweder:

- `uv run --extra modal modal setup`
- oder `MODAL_TOKEN_ID` und `MODAL_TOKEN_SECRET` in repo-root `.env`

## Smoke Test

```powershell
.\scripts\modal_mwe\smoke_test.ps1
```

Das Script baut ein kleines Testset und nutzt dann den echten Paketpfad:

```powershell
uv run --extra modal ads-and infer --backend modal ...
uv run --extra modal ads-and cost --output-dir ...
```
