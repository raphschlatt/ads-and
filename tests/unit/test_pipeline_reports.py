import json
from pathlib import Path

import pandas as pd

from author_name_disambiguation.common.pipeline_reports import (
    build_cluster_qc,
    build_infer_stage_metrics,
    build_pairs_qc,
    build_train_stage_metrics,
    write_compare_infer_to_baseline,
    write_compare_to_baseline,
    write_compare_train_to_baseline,
    write_json,
)


def _mentions(prefix: str, count: int, with_orcid: bool = False) -> pd.DataFrame:
    rows = []
    for i in range(count):
        row = {
            "mention_id": f"{prefix}{i}::0",
            "bibcode": f"{prefix}bib{i}",
            "author_idx": 0,
            "author_raw": f"Author {i}",
            "title": f"Title {i}",
            "abstract": f"Abstract {i}",
            "year": 2000 + (i % 20),
            "source_type": prefix,
            "block_key": f"{prefix}.blk{i // 2}",
        }
        if with_orcid:
            row["orcid"] = f"o{i // 2}"
            row["split"] = ["train", "val", "test", "train"][i % 4]
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_pairs_qc_and_cluster_qc():
    lspo_mentions = _mentions("lspo", 4, with_orcid=True)
    lspo_pairs = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "mention_id_1": "lspo0::0",
                "mention_id_2": "lspo1::0",
                "block_key": "lspo.blk0",
                "split": "val",
                "label": 0,
            },
            {
                "pair_id": "p2",
                "mention_id_1": "lspo2::0",
                "mention_id_2": "lspo3::0",
                "block_key": "lspo.blk1",
                "split": "test",
                "label": 1,
            },
        ]
    )
    ads_pairs = pd.DataFrame(
        [
            {
                "pair_id": "a1",
                "mention_id_1": "ads0::0",
                "mention_id_2": "ads1::0",
                "block_key": "ads.blk0",
                "split": "inference",
            }
        ]
    )
    qc = build_pairs_qc(
        lspo_mentions=lspo_mentions,
        lspo_pairs=lspo_pairs,
        ads_pairs=ads_pairs,
        split_meta={"status": "ok"},
    )

    assert qc["lspo_pairs"] == 2
    assert qc["ads_pairs"] == 1
    assert qc["split_balance"]["status"] == "ok"
    assert isinstance(qc["split_label_counts"], list)

    clusters = pd.DataFrame(
        [
            {"mention_id": "ads0::0", "block_key": "ads.blk0", "author_uid": "ads.blk0::0", "author_uid_local": "L0"},
            {"mention_id": "ads1::0", "block_key": "ads.blk0", "author_uid": "ads.blk0::1", "author_uid_local": "L1"},
        ]
    )
    pair_scores = pd.DataFrame(
        [
            {
                "pair_id": "a1",
                "mention_id_1": "ads0::0",
                "mention_id_2": "ads1::0",
                "block_key": "ads.blk0",
                "cosine_sim": 0.95,
                "distance": 0.05,
            }
        ]
    )
    cluster_qc = build_cluster_qc(pair_scores=pair_scores, clusters=clusters, threshold=0.80)

    assert cluster_qc["cluster_count"] == 2
    assert cluster_qc["split_high_sim_count"] == 1
    assert cluster_qc["split_high_sim_rate"] == 1.0
    assert cluster_qc["merged_low_conf_count"] == 0
    assert cluster_qc["merged_low_conf_rate"] == 0.0
    assert cluster_qc["probe_threshold"] == 0.35
    assert cluster_qc["split_high_sim_count_probe"] == 1
    assert cluster_qc["split_high_sim_rate_probe"] == 1.0
    assert cluster_qc["merged_low_conf_count_probe"] == 0
    assert cluster_qc["merged_low_conf_rate_probe"] == 0.0
    assert cluster_qc["n_pairs_evaluated"] == 1
    assert cluster_qc["pair_score_range_ok"] is True
    assert cluster_qc["negative_distance_count"] == 0
    assert cluster_qc["cosine_out_of_range_count"] == 0

    cluster_qc_local = build_cluster_qc(
        pair_scores=pair_scores,
        clusters=clusters,
        threshold=0.80,
        cluster_uid_col="author_uid_local",
    )
    assert cluster_qc_local["cluster_count"] == 2
    assert cluster_qc_local["split_high_sim_count"] == 1
    assert cluster_qc_local["split_high_sim_rate_probe"] == 1.0


def test_build_train_metrics_and_compare(tmp_path: Path):
    run_id = "smoke_current"
    baseline_run_id = "smoke_baseline"
    metrics_root = tmp_path / "metrics"
    current_dir = metrics_root / run_id
    baseline_dir = metrics_root / baseline_run_id
    current_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    lspo_mentions = _mentions("lspo", 4, with_orcid=True)
    train_manifest = {
        "best_val_f1": 0.91,
        "best_test_f1": 0.89,
        "best_threshold": 0.12,
        "best_threshold_selection_status": "ok",
        "best_threshold_source": "val_f1_opt",
        "best_val_class_counts": {"pos": 20, "neg": 5},
        "best_test_class_counts": {"pos": 18, "neg": 4},
        "precision_mode": "fp32",
    }
    split_meta = {"status": "ok", "max_possible_neg_total": 1234, "required_neg_total": 400}
    eps_meta = {
        "boundary_hit": True,
        "boundary_side": "max",
        "n_valid_candidates": 7,
        "f1_gap_best_second": 0.0123,
        "boundary_diagnostic_run": True,
        "range_limited": True,
        "diag_best_minus_canonical_f1": 0.021,
    }

    consistency_files = []
    for i in range(6):
        p = current_dir / f"{i:02d}_run_consistency.json"
        write_json({"run_id": run_id}, p)
        consistency_files.append(p)

    determinism_paths = [tmp_path / "lspo_manifest.parquet"]
    determinism_paths[0].write_text("x", encoding="utf-8")

    stage_metrics = build_train_stage_metrics(
        run_id=run_id,
        run_stage="smoke",
        lspo_mentions=lspo_mentions,
        train_manifest=train_manifest,
        consistency_files=consistency_files,
        determinism_paths=determinism_paths,
        split_meta=split_meta,
        eps_meta=eps_meta,
        subset_cache_key="smoke_seed11_target5000_cfg123_srcabc",
        lspo_pairs_count=42,
    )
    assert stage_metrics["metric_scope"] == "train"
    assert stage_metrics["schema_valid"] is True
    assert stage_metrics["run_id_consistent"] is True
    assert stage_metrics["determinism_valid"] is True
    assert stage_metrics["lspo_pairwise_f1"] == 0.89
    assert stage_metrics["lspo_pairwise_f1_val"] == 0.91
    assert stage_metrics["split_balance_status"] == "ok"
    assert stage_metrics["pair_score_range_ok"] is None
    assert stage_metrics["eps_boundary_hit"] is True
    assert stage_metrics["eps_boundary_side"] == "max"
    assert stage_metrics["eps_diag_ran"] is True
    assert stage_metrics["eps_range_limited"] is True
    assert stage_metrics["eps_diag_delta_f1"] == 0.021
    assert stage_metrics["precision_mode"] == "fp32"
    assert stage_metrics["subset_cache_key"] == "smoke_seed11_target5000_cfg123_srcabc"
    assert stage_metrics["lspo_pairs"] == 42
    assert stage_metrics["max_possible_neg_total"] == 1234
    assert stage_metrics["required_neg_total"] == 400
    assert stage_metrics["counts"]["ads_clusters"] == 0
    assert stage_metrics["counts"]["ads_cluster_assignments"] == 0
    assert stage_metrics["counts"]["ads_blocks"] == 0

    write_json(stage_metrics, current_dir / "05_stage_metrics_smoke.json")
    write_json({"go": True, "blockers": [], "warnings": ["eps_boundary_hit"]}, current_dir / "05_go_no_go_smoke.json")
    write_json(
        {
            "status": "ok",
            "split_label_counts": {"val": {"neg": 7}, "test": {"neg": 6}},
        },
        current_dir / "02_split_balance.json",
    )
    write_json(
        {
            "metric_scope": "train",
            "lspo_pairwise_f1": 0.88,
            "lspo_pairwise_f1_val": 0.90,
            "lspo_pairs": 40,
            "lspo_block_size_p95": 2.5,
            "counts": {"ads_clusters": 0},
        },
        baseline_dir / "05_stage_metrics_smoke.json",
    )
    write_json({"go": False, "warnings": []}, baseline_dir / "05_go_no_go_smoke.json")
    write_json(
        {
            "status": "split_balance_degraded",
            "split_label_counts": {"val": {"neg": 1}, "test": {"neg": 0}},
        },
        baseline_dir / "02_split_balance.json",
    )

    out = current_dir / "99_compare_train_to_baseline.json"
    write_compare_train_to_baseline(
        baseline_run_id=baseline_run_id,
        current_run_id=run_id,
        run_stage="smoke",
        metrics_root=metrics_root,
        output_path=out,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["compare_scope"] == "train"
    assert payload["split_status_current"] == "ok"
    assert payload["go_current"] is True
    assert payload["warnings_current"] == ["eps_boundary_hit"]
    assert payload["f1_delta"] == 0.010000000000000009


def test_build_infer_metrics_and_compare(tmp_path: Path):
    run_id = "infer_current"
    baseline_run_id = "infer_baseline"
    metrics_root = tmp_path / "metrics"
    current_dir = metrics_root / run_id
    baseline_dir = metrics_root / baseline_run_id
    current_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    ads_mentions = _mentions("ads", 4, with_orcid=False)
    clusters = pd.DataFrame(
        [
            {"mention_id": "ads0::0", "block_key": "ads.blk0", "author_uid": "g0", "author_uid_local": "l0"},
            {"mention_id": "ads1::0", "block_key": "ads.blk0", "author_uid": "g0", "author_uid_local": "l1"},
            {"mention_id": "ads2::0", "block_key": "ads.blk1", "author_uid": "g1", "author_uid_local": "l2"},
            {"mention_id": "ads3::0", "block_key": "ads.blk1", "author_uid": "g2", "author_uid_local": "l3"},
        ]
    )
    cluster_qc = {
        "pair_score_range_ok": True,
        "singleton_ratio": 0.20,
        "split_high_sim_rate": 0.11,
        "split_high_sim_rate_probe": 0.13,
        "merged_low_conf_rate": 0.07,
        "merged_low_conf_rate_probe": 0.09,
    }
    eps_meta = {
        "boundary_hit": False,
        "boundary_side": None,
        "n_valid_candidates": None,
        "f1_gap_best_second": None,
        "boundary_diagnostic_run": False,
        "range_limited": False,
        "diag_best_minus_canonical_f1": None,
    }
    consistency_files = []
    for i in [0, 1, 4, 5]:
        p = current_dir / f"{i:02d}_run_consistency.json"
        write_json({"run_id": run_id}, p)
        consistency_files.append(p)
    determinism_paths = [tmp_path / "ads_mentions.parquet", tmp_path / "scores.parquet", tmp_path / "clusters.parquet"]
    for p in determinism_paths:
        p.write_text("x", encoding="utf-8")

    stage_metrics = build_infer_stage_metrics(
        run_id=run_id,
        run_stage="infer_ads",
        ads_mentions=ads_mentions,
        clusters=clusters,
        consistency_files=consistency_files,
        determinism_paths=determinism_paths,
        cluster_qc=cluster_qc,
        eps_meta=eps_meta,
        threshold=0.31,
        threshold_selection_status="model_run_threshold",
        threshold_source="model_run",
        precision_mode="fp32",
    )
    assert stage_metrics["metric_scope"] == "infer"
    assert stage_metrics["counts"]["ads_clusters"] == 4
    assert stage_metrics["counts"]["ads_clusters_global_uid"] == 3
    assert stage_metrics["counts"]["ads_cluster_assignments"] == 4
    assert stage_metrics["counts"]["ads_blocks"] == 2
    assert stage_metrics["uid_local_to_global_valid"] is True
    assert stage_metrics["uid_local_to_global_max_nunique"] == 1
    assert stage_metrics["uid_global_to_local_max_nunique"] == 2
    assert stage_metrics["pair_score_range_ok"] is True
    assert stage_metrics["split_high_sim_rate_probe"] == 0.13
    assert stage_metrics["lspo_pairwise_f1"] is None

    write_json(stage_metrics, current_dir / "05_stage_metrics_infer_ads.json")
    write_json({"go": True, "blockers": [], "warnings": []}, current_dir / "05_go_no_go_infer_ads.json")
    write_json(
        {
            "metric_scope": "infer",
            "singleton_ratio": 0.10,
            "split_high_sim_rate_probe": 0.20,
            "merged_low_conf_rate_probe": 0.05,
            "counts": {"ads_mentions": 4, "ads_clusters": 2, "ads_cluster_assignments": 4},
        },
        baseline_dir / "05_stage_metrics_infer_ads.json",
    )
    write_json({"go": False, "warnings": []}, baseline_dir / "05_go_no_go_infer_ads.json")

    out = current_dir / "99_compare_infer_to_baseline.json"
    write_compare_infer_to_baseline(
        baseline_run_id=baseline_run_id,
        current_run_id=run_id,
        run_stage="infer_ads",
        metrics_root=metrics_root,
        output_path=out,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["compare_scope"] == "infer"
    assert payload["go_current"] is True
    assert payload["ads_clusters_baseline"] == 2
    assert payload["ads_clusters_current"] == 4
    assert payload["ads_clusters_delta"] == 2.0
    assert round(float(payload["split_high_sim_rate_probe_delta"]), 6) == -0.07


def test_legacy_compare_dispatches_to_scope_specific_writer(tmp_path: Path):
    metrics_root = tmp_path / "metrics"
    current_dir = metrics_root / "cur"
    baseline_dir = metrics_root / "base"
    current_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    write_json({"metric_scope": "infer", "counts": {"ads_clusters": 2}}, current_dir / "05_stage_metrics_infer_ads.json")
    write_json({"counts": {"ads_clusters": 1}}, baseline_dir / "05_stage_metrics_infer_ads.json")
    write_json({"go": True}, current_dir / "05_go_no_go_infer_ads.json")
    write_json({"go": False}, baseline_dir / "05_go_no_go_infer_ads.json")

    out = current_dir / "legacy_compare.json"
    write_compare_to_baseline(
        baseline_run_id="base",
        current_run_id="cur",
        run_stage="infer_ads",
        metrics_root=metrics_root,
        output_path=out,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["compare_scope"] == "infer"
