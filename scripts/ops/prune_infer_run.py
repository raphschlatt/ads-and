#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path
from typing import Final


DEFAULT_MODE: Final[str] = "json-only"
METADATA_KEEP_EXACT: Final[set[str]] = {
    "00_context.json",
    "01_input_summary.json",
    "02_preflight_infer.json",
    "03_pairs_qc.json",
    "summary.json",
}
METADATA_KEEP_PATTERNS: Final[tuple[str, ...]] = (
    "04_*",
    "05_*",
    "98_*",
    "99_compare_infer_*.json",
    "*_run_consistency.json",
)
PRODUCT_KEEP_FILES: Final[set[str]] = {
    "publications_disambiguated.parquet",
    "references_disambiguated.parquet",
    "source_author_assignments.parquet",
    "author_entities.parquet",
    "mention_clusters.parquet",
}
REQUIRED_RESOLUTION_FILES: Final[tuple[str, ...]] = (
    "98_infer_baseline_decision.json",
    "98_infer_baseline_decision.md",
    "99_compare_infer_to_baseline.json",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prune a resolved infer run directory while keeping comparison/decision metadata. "
            "This command only touches the selected run directory and never deletes training data."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Run directory under artifacts/exports.")
    parser.add_argument(
        "--mode",
        choices=("json-only", "product-only"),
        default=DEFAULT_MODE,
        help=(
            "Retention mode. 'json-only' keeps only small top-level metadata. "
            "'product-only' also keeps the final top-level product parquets."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without changing files.",
    )
    return parser


def _resolve_run_dir(raw: str) -> Path:
    run_dir = Path(str(raw)).expanduser()
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    return run_dir.resolve()


def _should_keep(name: str, *, mode: str) -> bool:
    if name in METADATA_KEEP_EXACT:
        return True
    if any(fnmatch.fnmatch(name, pattern) for pattern in METADATA_KEEP_PATTERNS):
        return True
    if mode == "product-only" and name in PRODUCT_KEEP_FILES:
        return True
    return False


def _validate_resolved_candidate(run_dir: Path, *, mode: str) -> None:
    missing = [name for name in REQUIRED_RESOLUTION_FILES if not (run_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            "Refusing to prune an unresolved candidate. "
            "Create compare/decision artifacts first: "
            f"{missing_text}"
        )
    if mode == "product-only":
        kept_products = [name for name in PRODUCT_KEEP_FILES if (run_dir / name).exists()]
        if not kept_products:
            raise SystemExit(
                "product-only mode requires at least one final top-level product parquet "
                f"in {run_dir}."
            )


def _remove_path(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = _resolve_run_dir(str(args.run_dir))
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    mode = str(args.mode)
    _validate_resolved_candidate(run_dir, mode=mode)

    removed_files: list[str] = []
    removed_dirs: list[str] = []
    kept_entries: list[str] = []

    for path in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if _should_keep(path.name, mode=mode):
            kept_entries.append(path.name)
            continue
        if path.is_dir() and not path.is_symlink():
            removed_dirs.append(path.name)
        else:
            removed_files.append(path.name)
        _remove_path(path, dry_run=bool(args.dry_run))

    payload = {
        "run_dir": str(run_dir),
        "mode": mode,
        "dry_run": bool(args.dry_run),
        "kept_entries": kept_entries,
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
