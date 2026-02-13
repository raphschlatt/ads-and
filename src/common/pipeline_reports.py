from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.common.io_schema import (
    CLUSTER_REQUIRED_COLUMNS,
    MENTION_REQUIRED_COLUMNS,
    validate_columns,
    validate_pair_score_ranges,
)


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)
    return p


def _safe_load_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _block_p95(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    block_sizes = df.groupby("block_key").size()
    if len(block_sizes) == 0:
        return 0.0
    return float(block_sizes.quantile(0.95))


def build_subset_summary(
    *,
    run_id: str,
    stage: str,
    source_fp: str,
    subset_tag: str,
    cache_hit: bool,
    lspo_subset: pd.DataFrame,
    ads_subset: pd.DataFrame,
    timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    timings = dict(timings or {})
    return {
        "run_id": str(run_id),
        "stage": str(stage),
        "source_fp": str(source_fp),
        "subset_tag": str(subset_tag),
        "cache_hit": bool(cache_hit),
        "read_lspo_s": float(timings.get("read_lspo_s", 0.0)),
        "read_ads_s": float(timings.get("read_ads_s", 0.0)),
        "build_lspo_s": float(timings.get("build_lspo_s", 0.0)),
        "build_ads_s": float(timings.get("build_ads_s", 0.0)),
        "save_lspo_shared_s": float(timings.get("save_lspo_shared_s", 0.0)),
        "save_ads_shared_s": float(timings.get("save_ads_shared_s", 0.0)),
        "save_lspo_run_s": float(timings.get("save_lspo_run_s", 0.0)),
        "save_ads_run_s": float(timings.get("save_ads_run_s", 0.0)),
        "total_s": float(timings.get("total_s", 0.0)),
        "lspo_mentions": int(len(lspo_subset)),
        "ads_mentions": int(len(ads_subset)),
        "lspo_blocks": int(lspo_subset["block_key"].nunique() if "block_key" in lspo_subset.columns else 0),
        "ads_blocks": int(ads_subset["block_key"].nunique() if "block_key" in ads_subset.columns else 0),
        "lspo_block_size_p95": float(_block_p95(lspo_subset)),
        "ads_block_size_p95": float(_block_p95(ads_subset)),
    }


def summarize_split_labels(pairs: pd.DataFrame) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for split in ["train", "val", "test", "inference", "mixed"]:
        sub = pairs[pairs["split"] == split] if "split" in pairs.columns else pairs.iloc[0:0]
        known = sub[sub["label"].notna()] if "label" in sub.columns else sub.iloc[0:0]
        rows.append(
            {
                "split": split,
                "pairs": int(len(sub)),
                "labeled_pairs": int(len(known)),
                "pos": int((known["label"] == 1).sum()) if len(known) else 0,
                "neg": int((known["label"] == 0).sum()) if len(known) else 0,
            }
        )
    return rows


def _orcid_leakage_groups(mentions: pd.DataFrame) -> int:
    if "orcid" not in mentions.columns or "split" not in mentions.columns:
        return 0
    g = mentions[mentions["orcid"].notna()].groupby("orcid")["split"].nunique()
    return int((g > 1).sum()) if len(g) else 0


def build_pairs_qc(
    *,
    lspo_mentions: pd.DataFrame,
    lspo_pairs: pd.DataFrame,
    ads_pairs: pd.DataFrame,
    split_meta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "orcid_leakage_groups": _orcid_leakage_groups(lspo_mentions),
        "lspo_pairs": int(len(lspo_pairs)),
        "ads_pairs": int(len(ads_pairs)),
        "split_label_counts": summarize_split_labels(lspo_pairs),
        "split_balance": dict(split_meta),
    }


def build_cluster_qc(
    *,
    pair_scores: pd.DataFrame,
    clusters: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    range_stats = validate_pair_score_ranges(pair_scores)

    if len(clusters) == 0:
        return {
            "singleton_ratio": 0.0,
            "merged_low_conf_count": 0,
            "split_high_sim_count": 0,
            "cluster_count": 0,
            **range_stats,
        }

    cluster_size = clusters.groupby(["block_key", "author_uid"]).size().rename("size").reset_index()
    singleton_ratio = float((cluster_size["size"] == 1).mean()) if len(cluster_size) else 0.0

    diag = pair_scores.merge(
        clusters[["mention_id", "author_uid"]].rename(
            columns={"mention_id": "mention_id_1", "author_uid": "author_uid_1"}
        ),
        on="mention_id_1",
        how="left",
    ).merge(
        clusters[["mention_id", "author_uid"]].rename(
            columns={"mention_id": "mention_id_2", "author_uid": "author_uid_2"}
        ),
        on="mention_id_2",
        how="left",
    )
    diag["same_cluster"] = diag["author_uid_1"] == diag["author_uid_2"]

    merged_low_conf_count = int(((diag["same_cluster"]) & (diag["cosine_sim"] < float(threshold))).sum())
    split_high_sim_count = int(((~diag["same_cluster"]) & (diag["cosine_sim"] >= float(threshold))).sum())

    return {
        "singleton_ratio": singleton_ratio,
        "merged_low_conf_count": merged_low_conf_count,
        "split_high_sim_count": split_high_sim_count,
        "cluster_count": int(len(cluster_size)),
        **range_stats,
    }


def _schema_valid(lspo_mentions: pd.DataFrame, ads_mentions: pd.DataFrame, clusters: pd.DataFrame) -> bool:
    try:
        validate_columns(lspo_mentions, MENTION_REQUIRED_COLUMNS, "lspo_mentions")
        validate_columns(ads_mentions, MENTION_REQUIRED_COLUMNS, "ads_mentions")
        validate_columns(clusters, CLUSTER_REQUIRED_COLUMNS, "clusters")
    except Exception:
        return False
    return True


def _run_id_consistent(run_id: str, consistency_files: Iterable[Path]) -> bool:
    expected = str(run_id)
    for p in consistency_files:
        data = _safe_load_json(p)
        if data is None:
            return False
        if str(data.get("run_id")) != expected:
            return False
    return True


def _determinism_valid(paths: Iterable[Path]) -> bool:
    return all(Path(p).exists() for p in paths)


def build_stage_metrics(
    *,
    run_id: str,
    run_stage: str,
    lspo_mentions: pd.DataFrame,
    ads_mentions: pd.DataFrame,
    clusters: pd.DataFrame,
    train_manifest: Mapping[str, Any],
    consistency_files: Iterable[Path],
    determinism_paths: Iterable[Path],
) -> dict[str, Any]:
    uid_uniqueness_max = (
        int(clusters.groupby("mention_id")["author_uid"].nunique().max()) if len(clusters) and "mention_id" in clusters.columns else 0
    )
    mention_coverage = (
        float(clusters["mention_id"].nunique()) / max(1, int(ads_mentions["mention_id"].nunique()))
        if len(ads_mentions) and "mention_id" in ads_mentions.columns and "mention_id" in clusters.columns
        else 0.0
    )

    lspo_pairwise_f1 = train_manifest.get("best_val_f1")
    lspo_pairwise_f1 = float(lspo_pairwise_f1) if lspo_pairwise_f1 is not None else None

    return {
        "run_id": str(run_id),
        "stage": str(run_stage),
        "schema_valid": _schema_valid(lspo_mentions=lspo_mentions, ads_mentions=ads_mentions, clusters=clusters),
        "determinism_valid": _determinism_valid(determinism_paths),
        "uid_uniqueness_valid": bool(uid_uniqueness_max <= 1),
        "uid_uniqueness_max": int(uid_uniqueness_max),
        "mention_coverage": float(mention_coverage),
        "run_id_consistent": _run_id_consistent(run_id, consistency_files),
        "lspo_pairwise_f1": lspo_pairwise_f1,
        "threshold": train_manifest.get("best_threshold"),
        "threshold_selection_status": train_manifest.get("best_threshold_selection_status", "unknown"),
        "threshold_source": train_manifest.get("best_threshold_source", "unknown"),
        "val_class_counts": train_manifest.get("best_val_class_counts", {}),
        "test_class_counts": train_manifest.get("best_test_class_counts", {}),
        "counts": {
            "lspo_mentions": int(len(lspo_mentions)),
            "ads_mentions": int(len(ads_mentions)),
            "ads_clusters": int(len(clusters)),
        },
    }


def write_compare_to_baseline(
    *,
    baseline_run_id: str,
    current_run_id: str,
    run_stage: str,
    metrics_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(metrics_root)
    baseline_dir = root / baseline_run_id
    current_dir = root / current_run_id

    baseline_stage = _safe_load_json(baseline_dir / f"05_stage_metrics_{run_stage}.json")
    current_stage = _safe_load_json(current_dir / f"05_stage_metrics_{run_stage}.json") or {}
    baseline_go = _safe_load_json(baseline_dir / f"05_go_no_go_{run_stage}.json")
    current_go = _safe_load_json(current_dir / f"05_go_no_go_{run_stage}.json") or {}
    baseline_split = _safe_load_json(baseline_dir / "02_split_balance.json")
    current_split = _safe_load_json(current_dir / "02_split_balance.json") or {}

    baseline_counts = ((baseline_split or {}).get("split_label_counts") or {})
    current_counts = (current_split.get("split_label_counts") or {})

    payload = {
        "baseline_run_id": str(baseline_run_id),
        "current_run_id": str(current_run_id),
        "baseline_stage_metrics_exists": bool(baseline_stage is not None),
        "split_status_baseline": (baseline_split or {}).get("status"),
        "split_status_current": current_split.get("status"),
        "val_neg_baseline": int(((baseline_counts.get("val") or {}).get("neg", 0))),
        "val_neg_current": int(((current_counts.get("val") or {}).get("neg", 0))),
        "test_neg_baseline": int(((baseline_counts.get("test") or {}).get("neg", 0))),
        "test_neg_current": int(((current_counts.get("test") or {}).get("neg", 0))),
        "f1_baseline": (baseline_stage or {}).get("lspo_pairwise_f1"),
        "f1_current": current_stage.get("lspo_pairwise_f1"),
        "go_baseline": (baseline_go or {}).get("go"),
        "go_current": current_go.get("go"),
        "blockers_current": current_go.get("blockers", []),
    }
    return write_json(payload, output_path)
