# Changelog

Notable changes to `ads-and`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Author strings are split only at semicolons. They were also split at commas,
  so `"Cole, Shaun; Lacey, Cedric"` became four authors instead of two.
- Rows with authors but without a `Bibcode` no longer abort the export. They
  stay in the output with empty `AuthorUID` and `AuthorDisplayName` lists and
  are counted as `authors_unmapped`. The same holds for a missing `Author`
  value in a string column, which was read as an author called `nan`.
- A rerun into an existing output directory returned the old results even for
  another input file, another `--infer-stage`, or another model. Outputs are
  now reused only if the request fingerprint in `summary.json` matches
  (publications, references, stage, model, dataset id and UID settings,
  cluster and gates config). Finished outputs of another request are neither
  returned nor overwritten: the run stops and asks for a new output directory
  or `--force`. This includes output directories written by earlier versions,
  which carry no fingerprint. The embedding caches of a run are checked
  against the names and texts they were built from.
- Duplicate records of one work (for example in publications and references)
  no longer pass identities on by author position alone. An author whose name
  differs from the first record's author at that position (another name block
  or a conflicting given name) stays unresolved
  (`assignment_kind=fallback_name_mismatch`). The run warns about it and
  reports the number as `authors_name_mismatch`.
- `psutil` and `scipy` are declared dependencies. Without `psutil`, Windows
  hosts had no RAM budget.

### Changed

- The README result table separates the bundled model (the training run with
  seed 5, 96.92% F1) from the mean of the five training runs (97.02%) and adds
  the trivial baseline in which every name block is one person (96.67%).
- README, `CITATION.cff`, and `.zenodo.json` describe author identifiers as
  valid within one run instead of stable.
- The `dev` and `research` extras include `tables`, which reads the LSPO HDF5
  release.
- The documentation marks run directories that are cited as evidence but are
  not tracked in Git, among them the ADS Modal run behind the cost and
  duration rule of thumb.
- `docs/lineage_and_modifications.md` was expanded: relation to NAND and LSPO,
  technical modifications, and where the reported metrics come from.

### Added

- Tests run on GitHub Actions for pushes to `main` and for pull requests, on
  Ubuntu and Windows.
- Tag-based release workflow with PyPI trusted publishing, release metadata
  checks, and Zenodo metadata (`.zenodo.json`).

## [0.1.3] - 2026-04-27

Releases up to 0.1.3 predate this changelog; see the Git tags `v0.1.1` to
`v0.1.3`.

[Unreleased]: https://github.com/raphschlatt/ads-and/compare/v0.1.3..HEAD
[0.1.3]: https://github.com/raphschlatt/ads-and/releases/tag/v0.1.3
