from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from author_name_disambiguation.common.io_schema import CLUSTER_REQUIRED_COLUMNS, MENTION_REQUIRED_COLUMNS, validate_columns


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)
    return p


def default_run_id(stage: str, *, tag: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_{ts}_{tag}{uuid.uuid4().hex[:8]}"


def _safe_load_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


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
    cache_key: str | None = None,
    cache_valid: bool | None = None,
    cache_invalid_reason: str | None = None,
    cache_rebuilt: bool = False,
    cache_version: str = "v3",
) -> dict[str, Any]:
    timings = dict(timings or {})
    return {
        "run_id": str(run_id),
        "stage": str(stage),
        "source_fp": str(source_fp),
        "subset_tag": str(subset_tag),
        "cache_key": str(cache_key) if cache_key is not None else str(subset_tag),
        "cache_hit": bool(cache_hit),
        "cache_valid": None if cache_valid is None else bool(cache_valid),
        "cache_invalid_reason": None if cache_invalid_reason is None else str(cache_invalid_reason),
        "cache_rebuilt": bool(cache_rebuilt),
        "cache_version": str(cache_version),
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
    lspo_pair_build_meta: Mapping[str, Any] | None = None,
    ads_pair_build_meta: Mapping[str, Any] | None = None,
    ads_pairs_count: int | None = None,
) -> dict[str, Any]:
    resolved_ads_pairs = int(ads_pairs_count) if ads_pairs_count is not None else int(len(ads_pairs))
    return {
        "orcid_leakage_groups": _orcid_leakage_groups(lspo_mentions),
        "lspo_pairs": int(len(lspo_pairs)),
        "ads_pairs": resolved_ads_pairs,
        "split_label_counts": summarize_split_labels(lspo_pairs),
        "split_balance": dict(split_meta),
        "lspo_pair_build": dict(lspo_pair_build_meta or {}),
        "ads_pair_build": dict(ads_pair_build_meta or {}),
    }


def build_cluster_qc(
    *,
    pair_scores: pd.DataFrame | str | Path,
    clusters: pd.DataFrame,
    threshold: float,
    probe_threshold: float = 0.35,
    chunk_rows: int = 200_000,
    cluster_uid_col: str = "author_uid",
) -> dict[str, Any]:
    if cluster_uid_col not in clusters.columns:
        raise ValueError(f"clusters missing required UID column for QC: {cluster_uid_col!r}")
    cluster_size = clusters.groupby(["block_key", cluster_uid_col]).size().rename("size").reset_index()
    singleton_ratio = float((cluster_size["size"] == 1).mean()) if len(cluster_size) else 0.0

    mention_to_uid = clusters.set_index("mention_id")[cluster_uid_col].astype(str).to_dict()

    cosine_min = None
    cosine_max = None
    distance_min = None
    distance_max = None
    cosine_non_finite_count = 0
    distance_non_finite_count = 0
    cosine_out_of_range_count = 0
    negative_distance_count = 0
    distance_above_max_count = 0
    n_pairs_evaluated = 0
    merged_low_conf_count = 0
    split_high_sim_count = 0
    merged_low_conf_count_probe = 0
    split_high_sim_count_probe = 0

    def _iter_chunks():
        if isinstance(pair_scores, pd.DataFrame):
            yield pair_scores
            return
        pair_path = Path(pair_scores)
        if not pair_path.exists():
            raise FileNotFoundError(pair_path)
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError("build_cluster_qc chunked mode requires pyarrow for parquet inputs.") from exc
        parquet = pq.ParquetFile(pair_path)
        for batch in parquet.iter_batches(batch_size=int(chunk_rows)):
            yield batch.to_pandas()

    for chunk in _iter_chunks():
        if len(chunk) == 0:
            continue
        n_pairs_evaluated += int(len(chunk))

        cosine = pd.to_numeric(chunk["cosine_sim"], errors="coerce")
        distance = pd.to_numeric(chunk["distance"], errors="coerce")
        cosine_non_finite_count += int(cosine.isna().sum())
        distance_non_finite_count += int(distance.isna().sum())

        cosine_finite = cosine[cosine.notna()]
        distance_finite = distance[distance.notna()]
        if len(cosine_finite):
            cmin = float(cosine_finite.min())
            cmax = float(cosine_finite.max())
            cosine_min = cmin if cosine_min is None else min(cosine_min, cmin)
            cosine_max = cmax if cosine_max is None else max(cosine_max, cmax)
            cosine_out_of_range_count += int(((cosine_finite < -1.0) | (cosine_finite > 1.0)).sum())
        if len(distance_finite):
            dmin = float(distance_finite.min())
            dmax = float(distance_finite.max())
            distance_min = dmin if distance_min is None else min(distance_min, dmin)
            distance_max = dmax if distance_max is None else max(distance_max, dmax)
            negative_distance_count += int((distance_finite < 0.0).sum())
            distance_above_max_count += int((distance_finite > 2.0).sum())

        uid1 = chunk["mention_id_1"].astype(str).map(mention_to_uid)
        uid2 = chunk["mention_id_2"].astype(str).map(mention_to_uid)
        same_cluster = uid1 == uid2
        cos = pd.to_numeric(chunk["cosine_sim"], errors="coerce")
        merged_low_conf_count += int((same_cluster & (cos < float(threshold))).sum())
        split_high_sim_count += int(((~same_cluster) & (cos >= float(threshold))).sum())
        merged_low_conf_count_probe += int((same_cluster & (cos < float(probe_threshold))).sum())
        split_high_sim_count_probe += int(((~same_cluster) & (cos >= float(probe_threshold))).sum())

    range_stats = {
        "pair_score_range_ok": bool(
            cosine_non_finite_count == 0
            and distance_non_finite_count == 0
            and cosine_out_of_range_count == 0
            and negative_distance_count == 0
            and distance_above_max_count == 0
        ),
        "cosine_min": cosine_min,
        "cosine_max": cosine_max,
        "distance_min": distance_min,
        "distance_max": distance_max,
        "cosine_non_finite_count": int(cosine_non_finite_count),
        "distance_non_finite_count": int(distance_non_finite_count),
        "cosine_out_of_range_count": int(cosine_out_of_range_count),
        "negative_distance_count": int(negative_distance_count),
        "distance_above_max_count": int(distance_above_max_count),
    }
    denom = max(1, n_pairs_evaluated)

    return {
        "singleton_ratio": singleton_ratio,
        "merged_low_conf_count": merged_low_conf_count,
        "merged_low_conf_rate": float(merged_low_conf_count / denom),
        "split_high_sim_count": split_high_sim_count,
        "split_high_sim_rate": float(split_high_sim_count / denom),
        "probe_threshold": float(probe_threshold),
        "merged_low_conf_count_probe": merged_low_conf_count_probe,
        "merged_low_conf_rate_probe": float(merged_low_conf_count_probe / denom),
        "split_high_sim_count_probe": split_high_sim_count_probe,
        "split_high_sim_rate_probe": float(split_high_sim_count_probe / denom),
        "n_pairs_evaluated": n_pairs_evaluated,
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


def _schema_valid_train(lspo_mentions: pd.DataFrame) -> bool:
    try:
        validate_columns(lspo_mentions, MENTION_REQUIRED_COLUMNS, "lspo_mentions")
    except Exception:
        return False
    return True


def _schema_valid_infer(ads_mentions: pd.DataFrame, clusters: pd.DataFrame) -> bool:
    try:
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


def build_train_stage_metrics(
    *,
    run_id: str,
    run_stage: str,
    lspo_mentions: pd.DataFrame,
    train_manifest: Mapping[str, Any],
    consistency_files: Iterable[Path],
    determinism_paths: Iterable[Path],
    split_meta: Mapping[str, Any] | None = None,
    eps_meta: Mapping[str, Any] | None = None,
    subset_cache_key: str | None = None,
    lspo_pairs_count: int | None = None,
) -> dict[str, Any]:
    split_meta = dict(split_meta or {})
    eps_meta = dict(eps_meta or {})

    lspo_pairwise_f1 = train_manifest.get("best_test_f1")
    lspo_pairwise_f1_source = "best_test_f1"
    if lspo_pairwise_f1 is None:
        lspo_pairwise_f1 = train_manifest.get("best_val_f1")
        lspo_pairwise_f1_source = "best_val_f1_legacy"
    lspo_pairwise_f1 = float(lspo_pairwise_f1) if lspo_pairwise_f1 is not None else None
    lspo_pairwise_f1_val = train_manifest.get("best_val_f1")
    lspo_pairwise_f1_val = float(lspo_pairwise_f1_val) if lspo_pairwise_f1_val is not None else None

    lspo_pairs_total = int(lspo_pairs_count) if lspo_pairs_count is not None else None
    split_counts = split_meta.get("split_label_counts") if isinstance(split_meta, Mapping) else None
    if lspo_pairs_total is None and isinstance(split_counts, Mapping):
        lspo_pairs_total = int(sum(int((split_counts.get(k) or {}).get("labeled_pairs", 0)) for k in ["train", "val", "test"]))

    lspo_blocks = int(lspo_mentions["block_key"].nunique()) if len(lspo_mentions) and "block_key" in lspo_mentions.columns else 0
    lspo_block_size_p95 = float(_block_p95(lspo_mentions))
    max_possible_neg_total = split_meta.get("max_possible_neg_total")
    required_neg_total = split_meta.get("required_neg_total")

    return {
        "run_id": str(run_id),
        "stage": str(run_stage),
        "metric_scope": "train",
        "schema_valid": _schema_valid_train(lspo_mentions=lspo_mentions),
        "determinism_valid": _determinism_valid(determinism_paths),
        "uid_uniqueness_valid": True,
        "uid_uniqueness_max": 1,
        "mention_coverage": None,
        "run_id_consistent": _run_id_consistent(run_id, consistency_files),
        "lspo_pairwise_f1": lspo_pairwise_f1,
        "lspo_pairwise_f1_source": lspo_pairwise_f1_source,
        "lspo_pairwise_f1_val": lspo_pairwise_f1_val,
        "threshold": train_manifest.get("best_threshold"),
        "threshold_selection_status": train_manifest.get("best_threshold_selection_status", "unknown"),
        "threshold_source": train_manifest.get("best_threshold_source", "unknown"),
        "precision_mode": train_manifest.get("precision_mode", "fp32"),
        "val_class_counts": train_manifest.get("best_val_class_counts", {}),
        "test_class_counts": train_manifest.get("best_test_class_counts", {}),
        "subset_cache_key": subset_cache_key,
        "lspo_pairs": lspo_pairs_total,
        "lspo_block_size_p95": lspo_block_size_p95,
        "max_possible_neg_total": max_possible_neg_total,
        "required_neg_total": required_neg_total,
        "split_balance_status": split_meta.get("status"),
        "pair_score_range_ok": None,
        "singleton_ratio": None,
        "split_high_sim_rate": None,
        "split_high_sim_rate_probe": None,
        "merged_low_conf_rate": None,
        "merged_low_conf_rate_probe": None,
        "eps_boundary_hit": bool(eps_meta.get("boundary_hit")) if "boundary_hit" in eps_meta else None,
        "eps_boundary_side": eps_meta.get("boundary_side"),
        "eps_n_valid_candidates": eps_meta.get("n_valid_candidates"),
        "eps_f1_gap_best_second": _optional_float(eps_meta.get("f1_gap_best_second")),
        "eps_diag_ran": bool(eps_meta.get("boundary_diagnostic_run")) if "boundary_diagnostic_run" in eps_meta else None,
        "eps_range_limited": bool(eps_meta.get("range_limited")) if "range_limited" in eps_meta else None,
        "eps_diag_delta_f1": _optional_float(eps_meta.get("diag_best_minus_canonical_f1")),
        "counts": {
            "lspo_mentions": int(len(lspo_mentions)),
            "lspo_blocks": lspo_blocks,
            "ads_mentions": 0,
            "ads_clusters": 0,
            "ads_cluster_assignments": 0,
            "ads_blocks": 0,
        },
    }


def build_infer_stage_metrics(
    *,
    run_id: str,
    run_stage: str,
    ads_mentions: pd.DataFrame,
    clusters: pd.DataFrame,
    consistency_files: Iterable[Path],
    determinism_paths: Iterable[Path],
    cluster_qc: Mapping[str, Any] | None = None,
    eps_meta: Mapping[str, Any] | None = None,
    threshold: float | None = None,
    threshold_selection_status: str = "model_run_threshold",
    threshold_source: str = "model_run",
    precision_mode: str = "fp32",
    infer_stage: str = "full",
    subset_tag: str | None = None,
    subset_ratio: float | None = None,
    memory_feasible: bool | None = None,
    pair_upper_bound: int | None = None,
    source_export_qc: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    precomputed_embeddings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cluster_qc = dict(cluster_qc or {})
    eps_meta = dict(eps_meta or {})
    source_export_qc = dict(source_export_qc or {})
    runtime = dict(runtime or {})
    precomputed_embeddings = dict(precomputed_embeddings or {})
    pubs_qc = dict(source_export_qc.get("publications", {}) or {})
    refs_qc = dict(source_export_qc.get("references", {}) or {})

    uid_uniqueness_max = (
        int(clusters.groupby("mention_id")["author_uid"].nunique().max()) if len(clusters) and "mention_id" in clusters.columns else 0
    )
    mention_coverage = (
        float(clusters["mention_id"].nunique()) / max(1, int(ads_mentions["mention_id"].nunique()))
        if len(ads_mentions) and "mention_id" in ads_mentions.columns and "mention_id" in clusters.columns
        else 0.0
    )
    ads_clusters_global_unique = (
        int(clusters["author_uid"].nunique()) if len(clusters) and "author_uid" in clusters.columns else 0
    )
    ads_clusters_local_unique = (
        int(clusters["author_uid_local"].nunique())
        if len(clusters) and "author_uid_local" in clusters.columns
        else ads_clusters_global_unique
    )
    ads_blocks = int(clusters["block_key"].nunique()) if len(clusters) and "block_key" in clusters.columns else 0

    uid_local_to_global_max_nunique: int | None = None
    uid_global_to_local_max_nunique: int | None = None
    uid_local_to_global_valid: bool | None = None
    if len(clusters) and "author_uid_local" in clusters.columns and "author_uid" in clusters.columns:
        local_to_global = clusters.groupby("author_uid_local")["author_uid"].nunique()
        global_to_local = clusters.groupby("author_uid")["author_uid_local"].nunique()
        uid_local_to_global_max_nunique = int(local_to_global.max()) if len(local_to_global) else 0
        uid_global_to_local_max_nunique = int(global_to_local.max()) if len(global_to_local) else 0
        uid_local_to_global_valid = bool(uid_local_to_global_max_nunique <= 1)

    return {
        "run_id": str(run_id),
        "stage": str(run_stage),
        "metric_scope": "infer",
        "schema_valid": _schema_valid_infer(ads_mentions=ads_mentions, clusters=clusters),
        "determinism_valid": _determinism_valid(determinism_paths),
        "uid_uniqueness_valid": bool(uid_uniqueness_max <= 1),
        "uid_uniqueness_max": int(uid_uniqueness_max),
        "uid_local_to_global_max_nunique": uid_local_to_global_max_nunique,
        "uid_global_to_local_max_nunique": uid_global_to_local_max_nunique,
        "uid_local_to_global_valid": uid_local_to_global_valid,
        "mention_coverage": float(mention_coverage),
        "run_id_consistent": _run_id_consistent(run_id, consistency_files),
        "lspo_pairwise_f1": None,
        "lspo_pairwise_f1_source": None,
        "lspo_pairwise_f1_val": None,
        "threshold": threshold,
        "threshold_selection_status": str(threshold_selection_status),
        "threshold_source": str(threshold_source),
        "precision_mode": str(precision_mode),
        "infer_stage": str(infer_stage),
        "subset_tag": subset_tag,
        "subset_ratio": _optional_float(subset_ratio),
        "memory_feasible": None if memory_feasible is None else bool(memory_feasible),
        "pair_upper_bound": None if pair_upper_bound is None else int(pair_upper_bound),
        "val_class_counts": {},
        "test_class_counts": {},
        "subset_cache_key": None,
        "lspo_pairs": None,
        "lspo_block_size_p95": None,
        "max_possible_neg_total": None,
        "required_neg_total": None,
        "split_balance_status": "not_applicable",
        "pair_score_range_ok": cluster_qc.get("pair_score_range_ok"),
        "singleton_ratio": _optional_float(cluster_qc.get("singleton_ratio")),
        "split_high_sim_rate": _optional_float(cluster_qc.get("split_high_sim_rate")),
        "split_high_sim_rate_probe": _optional_float(cluster_qc.get("split_high_sim_rate_probe")),
        "merged_low_conf_rate": _optional_float(cluster_qc.get("merged_low_conf_rate")),
        "merged_low_conf_rate_probe": _optional_float(cluster_qc.get("merged_low_conf_rate_probe")),
        "eps_boundary_hit": bool(eps_meta.get("boundary_hit")) if "boundary_hit" in eps_meta else None,
        "eps_boundary_side": eps_meta.get("boundary_side"),
        "eps_n_valid_candidates": eps_meta.get("n_valid_candidates"),
        "eps_f1_gap_best_second": _optional_float(eps_meta.get("f1_gap_best_second")),
        "eps_diag_ran": bool(eps_meta.get("boundary_diagnostic_run")) if "boundary_diagnostic_run" in eps_meta else None,
        "eps_range_limited": bool(eps_meta.get("range_limited")) if "range_limited" in eps_meta else None,
        "eps_diag_delta_f1": _optional_float(eps_meta.get("diag_best_minus_canonical_f1")),
        "counts": {
            "lspo_mentions": 0,
            "lspo_blocks": 0,
            "ads_mentions": int(len(ads_mentions)),
            "ads_clusters": ads_clusters_local_unique,
            "ads_clusters_global_uid": ads_clusters_global_unique,
            "ads_cluster_assignments": int(len(clusters)),
            "ads_blocks": ads_blocks,
        },
        "source_export": {
            "coverage_rate": _optional_float(source_export_qc.get("coverage_rate")),
            "authors_total": source_export_qc.get("authors_total"),
            "authors_mapped": source_export_qc.get("authors_mapped"),
            "authors_unmapped": source_export_qc.get("authors_unmapped"),
            "publications_coverage_rate": _optional_float(pubs_qc.get("coverage_rate")),
            "references_coverage_rate": _optional_float(refs_qc.get("coverage_rate")),
            "references_present": bool(source_export_qc.get("references_present")) if source_export_qc else None,
        },
        "runtime": runtime,
        "precomputed_embeddings": precomputed_embeddings,
    }


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
    cluster_qc: Mapping[str, Any] | None = None,
    split_meta: Mapping[str, Any] | None = None,
    eps_meta: Mapping[str, Any] | None = None,
    subset_cache_key: str | None = None,
    lspo_pairs_count: int | None = None,
) -> dict[str, Any]:
    cluster_qc = dict(cluster_qc or {})
    split_meta = dict(split_meta or {})
    eps_meta = dict(eps_meta or {})

    uid_uniqueness_max = (
        int(clusters.groupby("mention_id")["author_uid"].nunique().max()) if len(clusters) and "mention_id" in clusters.columns else 0
    )
    mention_coverage = (
        float(clusters["mention_id"].nunique()) / max(1, int(ads_mentions["mention_id"].nunique()))
        if len(ads_mentions) and "mention_id" in ads_mentions.columns and "mention_id" in clusters.columns
        else 0.0
    )
    ads_clusters_unique = (
        int(clusters["author_uid"].nunique()) if len(clusters) and "author_uid" in clusters.columns else 0
    )
    ads_blocks = int(clusters["block_key"].nunique()) if len(clusters) and "block_key" in clusters.columns else 0

    lspo_pairwise_f1 = train_manifest.get("best_test_f1")
    lspo_pairwise_f1_source = "best_test_f1"
    if lspo_pairwise_f1 is None:
        lspo_pairwise_f1 = train_manifest.get("best_val_f1")
        lspo_pairwise_f1_source = "best_val_f1_legacy"
    lspo_pairwise_f1 = float(lspo_pairwise_f1) if lspo_pairwise_f1 is not None else None
    lspo_pairwise_f1_val = train_manifest.get("best_val_f1")
    lspo_pairwise_f1_val = float(lspo_pairwise_f1_val) if lspo_pairwise_f1_val is not None else None
    lspo_pairs_total = int(lspo_pairs_count) if lspo_pairs_count is not None else None
    split_counts = split_meta.get("split_label_counts") if isinstance(split_meta, Mapping) else None
    if lspo_pairs_total is None and isinstance(split_counts, Mapping):
        lspo_pairs_total = int(sum(int((split_counts.get(k) or {}).get("labeled_pairs", 0)) for k in ["train", "val", "test"]))
    lspo_block_size_p95 = float(_block_p95(lspo_mentions))
    max_possible_neg_total = split_meta.get("max_possible_neg_total")
    required_neg_total = split_meta.get("required_neg_total")

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
        "lspo_pairwise_f1_source": lspo_pairwise_f1_source,
        "lspo_pairwise_f1_val": lspo_pairwise_f1_val,
        "threshold": train_manifest.get("best_threshold"),
        "threshold_selection_status": train_manifest.get("best_threshold_selection_status", "unknown"),
        "threshold_source": train_manifest.get("best_threshold_source", "unknown"),
        "precision_mode": train_manifest.get("precision_mode", "fp32"),
        "val_class_counts": train_manifest.get("best_val_class_counts", {}),
        "test_class_counts": train_manifest.get("best_test_class_counts", {}),
        "subset_cache_key": subset_cache_key,
        "lspo_pairs": lspo_pairs_total,
        "lspo_block_size_p95": lspo_block_size_p95,
        "max_possible_neg_total": max_possible_neg_total,
        "required_neg_total": required_neg_total,
        "split_balance_status": split_meta.get("status"),
        "pair_score_range_ok": cluster_qc.get("pair_score_range_ok"),
        "singleton_ratio": _optional_float(cluster_qc.get("singleton_ratio")),
        "split_high_sim_rate": _optional_float(cluster_qc.get("split_high_sim_rate")),
        "split_high_sim_rate_probe": _optional_float(cluster_qc.get("split_high_sim_rate_probe")),
        "merged_low_conf_rate": _optional_float(cluster_qc.get("merged_low_conf_rate")),
        "merged_low_conf_rate_probe": _optional_float(cluster_qc.get("merged_low_conf_rate_probe")),
        "eps_boundary_hit": bool(eps_meta.get("boundary_hit")) if "boundary_hit" in eps_meta else None,
        "eps_boundary_side": eps_meta.get("boundary_side"),
        "eps_n_valid_candidates": eps_meta.get("n_valid_candidates"),
        "eps_f1_gap_best_second": _optional_float(eps_meta.get("f1_gap_best_second")),
        "eps_diag_ran": bool(eps_meta.get("boundary_diagnostic_run")) if "boundary_diagnostic_run" in eps_meta else None,
        "eps_range_limited": bool(eps_meta.get("range_limited")) if "range_limited" in eps_meta else None,
        "eps_diag_delta_f1": _optional_float(eps_meta.get("diag_best_minus_canonical_f1")),
        "counts": {
            "lspo_mentions": int(len(lspo_mentions)),
            "ads_mentions": int(len(ads_mentions)),
            "ads_clusters": ads_clusters_unique,
            "ads_cluster_assignments": int(len(clusters)),
            "ads_blocks": ads_blocks,
        },
    }


def _delta(current_value: Any, baseline_value: Any) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    try:
        return float(current_value) - float(baseline_value)
    except Exception:
        return None


def _resolve_compare_dir(root: Path, run_ref: str | Path) -> Path:
    candidate = Path(run_ref)
    if candidate.exists():
        return candidate.resolve()
    return (root / str(run_ref)).resolve()


def _load_cluster_compare_frame(path: Path) -> tuple[pd.DataFrame | None, str | None, str | None]:
    if not path.exists():
        return None, None, "missing_artifact"
    try:
        sample = pd.read_parquet(path, columns=["mention_id", "block_key"])
    except Exception as exc:
        return None, None, f"read_failed:{exc!r}"

    uid_col = "author_uid_local" if "author_uid_local" in sample.columns else None
    if uid_col is None:
        try:
            probe = pd.read_parquet(path, columns=["author_uid_local"])
            uid_col = "author_uid_local" if "author_uid_local" in probe.columns else None
        except Exception:
            uid_col = None
    if uid_col is None:
        try:
            probe = pd.read_parquet(path, columns=["author_uid"])
            uid_col = "author_uid" if "author_uid" in probe.columns else None
        except Exception:
            uid_col = None
    if uid_col is None:
        return None, None, "missing_uid_column"

    try:
        frame = pd.read_parquet(path, columns=["mention_id", "block_key", uid_col])
    except Exception as exc:
        return None, None, f"read_failed:{exc!r}"
    return frame, uid_col, None


def _partition_compare_clusters(
    *,
    baseline_dir: Path,
    current_dir: Path,
) -> dict[str, Any]:
    baseline_frame, baseline_uid_col, baseline_err = _load_cluster_compare_frame(baseline_dir / "mention_clusters.parquet")
    current_frame, current_uid_col, current_err = _load_cluster_compare_frame(current_dir / "mention_clusters.parquet")
    if baseline_err is not None or current_err is not None:
        return {
            "status": "unavailable",
            "baseline_error": baseline_err,
            "current_error": current_err,
            "uid_column": None,
            "total_mentions_compared": None,
            "changed_mentions": None,
            "changed_blocks": None,
            "missing_in_current": None,
            "missing_in_baseline": None,
            "top_changed_blocks": [],
        }

    uid_col = baseline_uid_col if baseline_uid_col == current_uid_col else "author_uid_local"
    if uid_col != baseline_uid_col or uid_col != current_uid_col:
        return {
            "status": "unavailable",
            "baseline_error": f"baseline_uid_column={baseline_uid_col}",
            "current_error": f"current_uid_column={current_uid_col}",
            "uid_column": None,
            "total_mentions_compared": None,
            "changed_mentions": None,
            "changed_blocks": None,
            "missing_in_current": None,
            "missing_in_baseline": None,
            "top_changed_blocks": [],
        }

    baseline_frame = baseline_frame.rename(columns={uid_col: "cluster_uid"})
    current_frame = current_frame.rename(columns={uid_col: "cluster_uid"})
    merged = baseline_frame.merge(
        current_frame,
        on="mention_id",
        how="outer",
        suffixes=("_baseline", "_current"),
        indicator=True,
    )
    missing_in_current = int((merged["_merge"] == "left_only").sum())
    missing_in_baseline = int((merged["_merge"] == "right_only").sum())

    common = merged[merged["_merge"] == "both"].copy()
    exact_drift = common[
        (common["block_key_baseline"] != common["block_key_current"])
        | (common["cluster_uid_baseline"] != common["cluster_uid_current"])
    ]
    candidate_blocks: set[str] = set()
    candidate_blocks.update(str(v) for v in exact_drift["block_key_baseline"].dropna().astype(str).tolist())
    candidate_blocks.update(str(v) for v in exact_drift["block_key_current"].dropna().astype(str).tolist())
    candidate_blocks.update(str(v) for v in merged.loc[merged["_merge"] == "left_only", "block_key_baseline"].dropna().astype(str).tolist())
    candidate_blocks.update(str(v) for v in merged.loc[merged["_merge"] == "right_only", "block_key_current"].dropna().astype(str).tolist())

    drift_blocks: list[dict[str, Any]] = []
    drift_mention_ids: set[str] = set()
    for block_key in sorted(candidate_blocks):
        baseline_block = baseline_frame[baseline_frame["block_key"].astype(str) == str(block_key)]
        current_block = current_frame[current_frame["block_key"].astype(str) == str(block_key)]
        baseline_parts = {frozenset(g["mention_id"].astype(str).tolist()) for _, g in baseline_block.groupby("cluster_uid")}
        current_parts = {frozenset(g["mention_id"].astype(str).tolist()) for _, g in current_block.groupby("cluster_uid")}
        if baseline_parts == current_parts:
            continue
        block_compare = baseline_block.merge(
            current_block,
            on="mention_id",
            how="outer",
            suffixes=("_baseline", "_current"),
            indicator=True,
        )
        changed_mentions = int(
            (
                (block_compare["_merge"] != "both")
                | (block_compare["cluster_uid_baseline"] != block_compare["cluster_uid_current"])
                | (block_compare["block_key_baseline"] != block_compare["block_key_current"])
            ).sum()
        )
        changed_mask = (
            (block_compare["_merge"] != "both")
            | (block_compare["cluster_uid_baseline"] != block_compare["cluster_uid_current"])
            | (block_compare["block_key_baseline"] != block_compare["block_key_current"])
        )
        drift_mention_ids.update(block_compare.loc[changed_mask, "mention_id"].dropna().astype(str).tolist())
        drift_blocks.append(
            {
                "block_key": str(block_key),
                "mentions": int(len(pd.unique(pd.concat([baseline_block["mention_id"], current_block["mention_id"]])))),
                "baseline_clusters": int(baseline_block["cluster_uid"].nunique()),
                "current_clusters": int(current_block["cluster_uid"].nunique()),
                "changed_mentions": int(changed_mentions),
            }
        )

    drift_blocks.sort(key=lambda row: (-int(row["changed_mentions"]), str(row["block_key"])))
    changed_mentions_total = int(len(drift_mention_ids))
    return {
        "status": "ok",
        "baseline_error": None,
        "current_error": None,
        "uid_column": str(uid_col),
        "total_mentions_compared": int(len(common)),
        "changed_mentions": int(changed_mentions_total),
        "changed_blocks": int(len(drift_blocks)),
        "missing_in_current": int(missing_in_current),
        "missing_in_baseline": int(missing_in_baseline),
        "top_changed_blocks": drift_blocks[:10],
    }


def write_compare_train_to_baseline(
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
    baseline_stage_counts = (baseline_stage or {}).get("counts") or {}
    current_stage_counts = current_stage.get("counts") or {}

    payload = {
        "compare_scope": "train",
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
        "f1_delta": _delta(current_stage.get("lspo_pairwise_f1"), (baseline_stage or {}).get("lspo_pairwise_f1")),
        "f1_val_baseline": (baseline_stage or {}).get("lspo_pairwise_f1_val"),
        "f1_val_current": current_stage.get("lspo_pairwise_f1_val"),
        "f1_val_delta": _delta(current_stage.get("lspo_pairwise_f1_val"), (baseline_stage or {}).get("lspo_pairwise_f1_val")),
        "lspo_pairs_baseline": (baseline_stage or {}).get("lspo_pairs"),
        "lspo_pairs_current": current_stage.get("lspo_pairs"),
        "lspo_pairs_delta": _delta(current_stage.get("lspo_pairs"), (baseline_stage or {}).get("lspo_pairs")),
        "lspo_block_size_p95_baseline": (baseline_stage or {}).get("lspo_block_size_p95"),
        "lspo_block_size_p95_current": current_stage.get("lspo_block_size_p95"),
        "lspo_block_size_p95_delta": _delta(
            current_stage.get("lspo_block_size_p95"),
            (baseline_stage or {}).get("lspo_block_size_p95"),
        ),
        "go_baseline": (baseline_go or {}).get("go"),
        "go_current": current_go.get("go"),
        "warnings_baseline": (baseline_go or {}).get("warnings", []),
        "warnings_current": current_go.get("warnings", []),
        "blockers_current": current_go.get("blockers", []),
        "ads_clusters_baseline": baseline_stage_counts.get("ads_clusters"),
        "ads_clusters_current": current_stage_counts.get("ads_clusters"),
    }
    return write_json(payload, output_path)


def write_compare_infer_to_baseline(
    *,
    baseline_run_id: str,
    current_run_id: str,
    run_stage: str,
    metrics_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(metrics_root)
    baseline_dir = _resolve_compare_dir(root, baseline_run_id)
    current_dir = _resolve_compare_dir(root, current_run_id)

    baseline_stage = _safe_load_json(baseline_dir / f"05_stage_metrics_{run_stage}.json")
    current_stage = _safe_load_json(current_dir / f"05_stage_metrics_{run_stage}.json") or {}
    baseline_go = _safe_load_json(baseline_dir / f"05_go_no_go_{run_stage}.json")
    current_go = _safe_load_json(current_dir / f"05_go_no_go_{run_stage}.json") or {}

    baseline_stage_counts = (baseline_stage or {}).get("counts") or {}
    current_stage_counts = current_stage.get("counts") or {}

    singleton_ratio_baseline = (baseline_stage or {}).get("singleton_ratio")
    singleton_ratio_current = current_stage.get("singleton_ratio")
    split_high_sim_rate_probe_baseline = (baseline_stage or {}).get("split_high_sim_rate_probe")
    split_high_sim_rate_probe_current = current_stage.get("split_high_sim_rate_probe")
    merged_low_conf_rate_probe_baseline = (baseline_stage or {}).get("merged_low_conf_rate_probe")
    merged_low_conf_rate_probe_current = current_stage.get("merged_low_conf_rate_probe")
    baseline_source_export = (baseline_stage or {}).get("source_export") or {}
    current_source_export = current_stage.get("source_export") or {}
    cluster_compare = _partition_compare_clusters(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
    )

    payload = {
        "compare_scope": "infer",
        "baseline_run_id": str(baseline_run_id),
        "current_run_id": str(current_run_id),
        "baseline_dir": str(baseline_dir),
        "current_dir": str(current_dir),
        "baseline_stage_run_id": (baseline_stage or {}).get("run_id"),
        "current_stage_run_id": current_stage.get("run_id"),
        "baseline_stage_metrics_exists": bool(baseline_stage is not None),
        "go_baseline": (baseline_go or {}).get("go"),
        "go_current": current_go.get("go"),
        "warnings_baseline": (baseline_go or {}).get("warnings", []),
        "warnings_current": current_go.get("warnings", []),
        "blockers_current": current_go.get("blockers", []),
        "ads_mentions_baseline": baseline_stage_counts.get("ads_mentions"),
        "ads_mentions_current": current_stage_counts.get("ads_mentions"),
        "ads_mentions_delta": _delta(current_stage_counts.get("ads_mentions"), baseline_stage_counts.get("ads_mentions")),
        "ads_clusters_baseline": baseline_stage_counts.get("ads_clusters"),
        "ads_clusters_current": current_stage_counts.get("ads_clusters"),
        "ads_clusters_delta": _delta(current_stage_counts.get("ads_clusters"), baseline_stage_counts.get("ads_clusters")),
        "ads_cluster_assignments_baseline": baseline_stage_counts.get("ads_cluster_assignments"),
        "ads_cluster_assignments_current": current_stage_counts.get("ads_cluster_assignments"),
        "ads_cluster_assignments_delta": _delta(
            current_stage_counts.get("ads_cluster_assignments"),
            baseline_stage_counts.get("ads_cluster_assignments"),
        ),
        "singleton_ratio_baseline": singleton_ratio_baseline,
        "singleton_ratio_current": singleton_ratio_current,
        "singleton_ratio_delta": _delta(singleton_ratio_current, singleton_ratio_baseline),
        "split_high_sim_rate_probe_baseline": split_high_sim_rate_probe_baseline,
        "split_high_sim_rate_probe_current": split_high_sim_rate_probe_current,
        "split_high_sim_rate_probe_delta": _delta(
            split_high_sim_rate_probe_current,
            split_high_sim_rate_probe_baseline,
        ),
        "merged_low_conf_rate_probe_baseline": merged_low_conf_rate_probe_baseline,
        "merged_low_conf_rate_probe_current": merged_low_conf_rate_probe_current,
        "merged_low_conf_rate_probe_delta": _delta(
            merged_low_conf_rate_probe_current,
            merged_low_conf_rate_probe_baseline,
        ),
        "infer_stage_baseline": (baseline_stage or {}).get("infer_stage"),
        "infer_stage_current": current_stage.get("infer_stage"),
        "subset_ratio_baseline": (baseline_stage or {}).get("subset_ratio"),
        "subset_ratio_current": current_stage.get("subset_ratio"),
        "subset_ratio_delta": _delta(current_stage.get("subset_ratio"), (baseline_stage or {}).get("subset_ratio")),
        "pair_upper_bound_baseline": (baseline_stage or {}).get("pair_upper_bound"),
        "pair_upper_bound_current": current_stage.get("pair_upper_bound"),
        "pair_upper_bound_delta": _delta(
            current_stage.get("pair_upper_bound"),
            (baseline_stage or {}).get("pair_upper_bound"),
        ),
        "source_coverage_rate_baseline": baseline_source_export.get("coverage_rate"),
        "source_coverage_rate_current": current_source_export.get("coverage_rate"),
        "source_coverage_rate_delta": _delta(
            current_source_export.get("coverage_rate"),
            baseline_source_export.get("coverage_rate"),
        ),
        "mention_cluster_compare_status": cluster_compare.get("status"),
        "mention_cluster_compare_uid_column": cluster_compare.get("uid_column"),
        "mention_cluster_compare_total_mentions": cluster_compare.get("total_mentions_compared"),
        "mention_cluster_changed_mentions": cluster_compare.get("changed_mentions"),
        "mention_cluster_changed_blocks": cluster_compare.get("changed_blocks"),
        "mention_cluster_missing_in_current": cluster_compare.get("missing_in_current"),
        "mention_cluster_missing_in_baseline": cluster_compare.get("missing_in_baseline"),
        "mention_cluster_top_changed_blocks": cluster_compare.get("top_changed_blocks", []),
        "mention_cluster_compare_baseline_error": cluster_compare.get("baseline_error"),
        "mention_cluster_compare_current_error": cluster_compare.get("current_error"),
    }
    return write_json(payload, output_path)


def write_compare_to_baseline(
    *,
    baseline_run_id: str,
    current_run_id: str,
    run_stage: str,
    metrics_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Legacy compare writer.

    This keeps older callers functional by selecting compare scope from current stage metrics.
    New code should call write_compare_train_to_baseline or write_compare_infer_to_baseline.
    """
    current_stage = _safe_load_json(Path(metrics_root) / current_run_id / f"05_stage_metrics_{run_stage}.json") or {}
    scope = str(current_stage.get("metric_scope", "")).strip().lower()
    if scope == "infer" or str(run_stage) == "infer_sources":
        return write_compare_infer_to_baseline(
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            run_stage=run_stage,
            metrics_root=metrics_root,
            output_path=output_path,
        )
    return write_compare_train_to_baseline(
        baseline_run_id=baseline_run_id,
        current_run_id=current_run_id,
        run_stage=run_stage,
        metrics_root=metrics_root,
        output_path=output_path,
    )
