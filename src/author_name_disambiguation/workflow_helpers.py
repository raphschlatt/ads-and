from __future__ import annotations

import re
from pathlib import Path

from author_name_disambiguation.common.pipeline_reports import default_run_id

_REPORT_TAG_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def default_train_run_id(stage: str) -> str:
    return default_run_id(stage, tag="cli")


def sanitize_report_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    value = str(tag).strip()
    if value == "":
        raise ValueError("report_tag must be non-empty when provided.")
    if _REPORT_TAG_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "Invalid report_tag. Allowed characters are: a-z, A-Z, 0-9, '.', '_' and '-'."
        )
    return value


def resolve_report_paths(metrics_dir: Path, report_tag: str | None) -> dict[str, Path]:
    suffix = "" if report_tag is None else f"__{report_tag}"
    return {
        "json": metrics_dir / f"06_clustering_test_report{suffix}.json",
        "summary_csv": metrics_dir / f"06_clustering_test_summary{suffix}.csv",
        "per_seed_csv": metrics_dir / f"06_clustering_test_per_seed{suffix}.csv",
        "markdown": metrics_dir / f"06_clustering_test_report{suffix}.md",
    }
