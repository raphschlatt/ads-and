#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml


def _fail(message: str) -> None:
    raise SystemExit(f"release check failed: {message}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        _fail(f"expected YAML mapping in {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        _fail("usage: check_release_version.py vX.Y.Z")

    tag = args[0].strip()
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        _fail(f"tag must look like vX.Y.Z, got {tag!r}")
    version = tag.removeprefix("v")

    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = str(pyproject.get("project", {}).get("version", "")).strip()
    if project_version != version:
        _fail(f"pyproject version {project_version!r} does not match tag {tag!r}")

    citation = _load_yaml(repo_root / "CITATION.cff")
    citation_version = str(citation.get("version", "")).strip()
    if citation_version != version:
        _fail(f"CITATION.cff version {citation_version!r} does not match tag {tag!r}")
    if not str(citation.get("date-released", "")).strip():
        _fail("CITATION.cff date-released is missing")

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    if "img.shields.io/pypi/v/ads_and" in readme or "img.shields.io/pypi/pyversions/ads_and" in readme:
        _fail("README uses non-canonical ads_and PyPI badge URL; use ads-and")
    for expected in ("img.shields.io/pypi/v/ads-and.svg", "img.shields.io/pypi/pyversions/ads-and.svg"):
        if expected not in readme:
            _fail(f"README missing canonical badge URL {expected!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
