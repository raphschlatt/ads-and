#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from author_name_disambiguation.features.embed_chars2vec import generate_chars2vec_embeddings


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_output_path(repo_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "artifacts" / "benchmarks" / f"chars2vec_modes_{timestamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark chars2vec execution modes on a newline-delimited names file.")
    parser.add_argument("--names-file", required=True, help="Text file with one author name per line.")
    parser.add_argument("--model-name", default="eng_50", help="chars2vec model name to load. Default: eng_50")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--quiet-libraries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silence noisy TensorFlow/chars2vec startup logs. Default: true",
    )
    parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pretty-print output JSON. Default: true",
    )
    return parser


def _load_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").splitlines()


def _run_case(
    *,
    label: str,
    names: list[str],
    model_name: str,
    batch_size: int | None,
    execution_mode: str,
    quiet_libraries: bool,
) -> dict[str, Any]:
    started_at = perf_counter()
    embeddings, meta = generate_chars2vec_embeddings(
        names=names,
        model_name=model_name,
        batch_size=batch_size,
        execution_mode=execution_mode,
        quiet_libraries=quiet_libraries,
        show_progress=False,
        return_meta=True,
    )
    wall_seconds = float(perf_counter() - started_at)
    return {
        "label": label,
        "execution_mode": execution_mode,
        "requested_batch_size": batch_size,
        "wall_seconds": wall_seconds,
        "embedding_shape": list(embeddings.shape),
        "meta": meta,
    }


def main() -> int:
    args = _build_parser().parse_args()
    names_file = Path(args.names_file)
    names = _load_names(names_file)
    if not names:
        raise ValueError("--names-file must contain at least one line.")

    cases = [
        {"label": "predict_32", "execution_mode": "predict", "batch_size": 32},
        {"label": "predict_auto", "execution_mode": "predict", "batch_size": None},
        {"label": "direct_call", "execution_mode": "direct_call", "batch_size": None},
    ]

    results = [
        _run_case(
            label=str(case["label"]),
            names=names,
            model_name=str(args.model_name),
            batch_size=case["batch_size"],
            execution_mode=str(case["execution_mode"]),
            quiet_libraries=bool(args.quiet_libraries),
        )
        for case in cases
    ]

    payload = {
        "created_at_utc": _utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "names_file": str(names_file),
        "model_name": str(args.model_name),
        "name_count": int(len(names)),
        "unique_name_count": int(len({str(name).lower() for name in names})),
        "quiet_libraries": bool(args.quiet_libraries),
        "cases": results,
    }

    output_path = Path(args.output) if args.output else _default_output_path(REPO_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
