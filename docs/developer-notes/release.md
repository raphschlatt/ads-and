# Release Runbook

Use GitHub Actions for releases. Do not publish locally except as an emergency
fallback.

## One-time setup

- In PyPI `ads-and` settings, add a Trusted Publisher:
  - owner: `raphschlatt`
  - repository: `ads-and`
  - workflow: `release.yml`
  - environment: `pypi`
- In GitHub, create environment `pypi`; require reviewer approval if desired.

## Normal release

1. Apply code/docs/artifact changes and run local checks.
2. If outputs may change, run the LSPO quality gate on a GPU host and commit the
   small release evidence files.
3. Update `pyproject.toml`, `uv.lock`, `CITATION.cff`, and `.zenodo.json`.
   Keep `CITATION.cff` on the Zenodo concept DOI; do not replace it with a
   version-specific Zenodo DOI.
4. Commit to `main`, then tag and push:

```powershell
git tag -a v0.1.4 -m "ads-and 0.1.4"
git push origin main
git push origin v0.1.4
```

The `Release` workflow validates metadata, checks baseline evidence, runs tests,
builds once, publishes the same artifacts to PyPI, and creates the GitHub
release. Keep README badges on the canonical PyPI name `ads-and`.
