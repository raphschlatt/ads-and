#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


DEFAULT_KEEP_FILES = (
    "00_context.json",
    "05_stage_metrics_infer_sources.json",
    "05_go_no_go_infer_sources.json",
    "99_compare_infer_to_baseline.json",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune an ADS infer candidate down to the JSON-only retention set."
    )
    parser.add_argument("--run-dir", required=True, help="Path to artifacts/exports/<run_id> directory.")
    parser.add_argument(
        "--keep-file",
        action="append",
        default=[],
        help="Additional file name inside the run directory to keep. Can be repeated.",
    )
    return parser


def _size_bytes(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += int(child.stat().st_size)
    return total


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    keep_names = set(DEFAULT_KEEP_FILES)
    keep_names.update(str(name).strip() for name in args.keep_file if str(name).strip())
    missing_keep = sorted(name for name in keep_names if not (run_dir / name).exists())
    if missing_keep:
        raise FileNotFoundError(
            f"Cannot prune {run_dir}: missing required keep files: {', '.join(missing_keep)}"
        )

    removed_files: list[str] = []
    removed_bytes = 0
    for child in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if child.name in keep_names:
            continue
        removed_bytes += _size_bytes(child)
        removed_files.append(child.name)
        _remove_path(child)

    payload = {
        "run_dir": str(run_dir),
        "kept_files": sorted(keep_names),
        "removed_files": removed_files,
        "removed_bytes": int(removed_bytes),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
