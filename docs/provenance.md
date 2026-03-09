# Provenance

This repository is the maintained package form of a larger research code lineage around neural author name disambiguation.

## What Changed

- The package namespace is now `author_name_disambiguation`.
- The public surface was reduced to four CLI commands and one source-based inference API.
- Notebook-specific runtime assumptions and repo-root heuristics were removed from the public package path.
- The vendored snapshot directory `neural_name_dismabiguator-main/` was removed from the repository tree.

## What Stays in Git History

- prior reconstruction steps
- research-driven refactors
- earlier experimental layouts and interfaces

The current package does not ship the old vendored directory as part of the installable product. Provenance is tracked through repository history and this documentation instead.
