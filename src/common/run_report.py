from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import pandas as pd


def load_stage_metrics(metrics_file: str | Path) -> Dict:
    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_go_no_go(stage_metrics: Dict) -> Dict:
    checks: List[Dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add_check(
        "schema_valid",
        stage_metrics.get("schema_valid", False),
        "All required schemas validated" if stage_metrics.get("schema_valid") else "Schema check failed",
    )
    add_check(
        "determinism_valid",
        stage_metrics.get("determinism_valid", False),
        "Determinism checks passed" if stage_metrics.get("determinism_valid") else "Determinism mismatch",
    )
    add_check(
        "uid_uniqueness",
        stage_metrics.get("uid_uniqueness_valid", False),
        "Each mention_id mapped to one author_uid" if stage_metrics.get("uid_uniqueness_valid") else "UID mapping issue",
    )

    # If LSPO metrics are present, apply loose stage-aware sanity bounds.
    lspo_f1 = stage_metrics.get("lspo_pairwise_f1")
    if lspo_f1 is not None:
        add_check("lspo_pairwise_f1_sanity", lspo_f1 >= 0.70, f"Observed F1={lspo_f1:.4f}")

    passed = all(c["passed"] for c in checks)
    return {"go": passed, "checks": checks}


def write_go_no_go_report(result: Dict, output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return p


def summarize_block_distribution(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    # pandas compatibility: older versions do not support reset_index(names=...)
    dist = df["block_key"].value_counts().rename_axis("block_key").reset_index(name="mention_count")
    return dist.head(top_k)
