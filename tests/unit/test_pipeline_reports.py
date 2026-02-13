from pathlib import Path

import pandas as pd

from src.common.pipeline_reports import (
    build_cluster_qc,
    build_pairs_qc,
    build_stage_metrics,
    write_compare_to_baseline,
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
            {"mention_id": "ads0::0", "block_key": "ads.blk0", "author_uid": "ads.blk0::0"},
            {"mention_id": "ads1::0", "block_key": "ads.blk0", "author_uid": "ads.blk0::1"},
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
    assert cluster_qc["merged_low_conf_count"] == 0
    assert cluster_qc["pair_score_range_ok"] is True
    assert cluster_qc["negative_distance_count"] == 0
    assert cluster_qc["cosine_out_of_range_count"] == 0


def test_stage_metrics_and_compare_to_baseline(tmp_path: Path):
    run_id = "smoke_current"
    baseline_run_id = "smoke_baseline"
    metrics_root = tmp_path / "metrics"
    current_dir = metrics_root / run_id
    baseline_dir = metrics_root / baseline_run_id
    current_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    lspo_mentions = _mentions("lspo", 4, with_orcid=True)
    ads_mentions = _mentions("ads", 4, with_orcid=False)
    clusters = pd.DataFrame(
        [
            {"mention_id": "ads0::0", "block_key": "ads.blk0", "author_uid": "ads.blk0::0"},
            {"mention_id": "ads1::0", "block_key": "ads.blk0", "author_uid": "ads.blk0::0"},
            {"mention_id": "ads2::0", "block_key": "ads.blk1", "author_uid": "ads.blk1::0"},
            {"mention_id": "ads3::0", "block_key": "ads.blk1", "author_uid": "ads.blk1::1"},
        ]
    )
    train_manifest = {
        "best_val_f1": 0.91,
        "best_threshold": 0.12,
        "best_threshold_selection_status": "ok",
        "best_threshold_source": "val_f1_opt",
        "best_val_class_counts": {"pos": 20, "neg": 5},
        "best_test_class_counts": {"pos": 18, "neg": 4},
    }

    consistency_files = []
    for i in range(6):
        p = current_dir / f"{i:02d}_run_consistency.json"
        write_json({"run_id": run_id}, p)
        consistency_files.append(p)

    determinism_paths = [tmp_path / "lspo_manifest.parquet", tmp_path / "ads_manifest.parquet"]
    determinism_paths[0].write_text("x", encoding="utf-8")
    determinism_paths[1].write_text("x", encoding="utf-8")

    stage_metrics = build_stage_metrics(
        run_id=run_id,
        run_stage="smoke",
        lspo_mentions=lspo_mentions,
        ads_mentions=ads_mentions,
        clusters=clusters,
        train_manifest=train_manifest,
        consistency_files=consistency_files,
        determinism_paths=determinism_paths,
    )
    assert stage_metrics["run_id"] == run_id
    assert stage_metrics["schema_valid"] is True
    assert stage_metrics["run_id_consistent"] is True
    assert stage_metrics["determinism_valid"] is True
    assert stage_metrics["lspo_pairwise_f1"] == 0.91

    write_json(stage_metrics, current_dir / "05_stage_metrics_smoke.json")
    write_json({"go": True, "blockers": []}, current_dir / "05_go_no_go_smoke.json")
    write_json(
        {
            "status": "ok",
            "split_label_counts": {"val": {"neg": 7}, "test": {"neg": 6}},
        },
        current_dir / "02_split_balance.json",
    )

    write_json({"lspo_pairwise_f1": 0.88}, baseline_dir / "05_stage_metrics_smoke.json")
    write_json({"go": False}, baseline_dir / "05_go_no_go_smoke.json")
    write_json(
        {
            "status": "split_balance_degraded",
            "split_label_counts": {"val": {"neg": 1}, "test": {"neg": 0}},
        },
        baseline_dir / "02_split_balance.json",
    )

    out = current_dir / "99_compare_to_baseline.json"
    write_compare_to_baseline(
        baseline_run_id=baseline_run_id,
        current_run_id=run_id,
        run_stage="smoke",
        metrics_root=metrics_root,
        output_path=out,
    )
    payload = out.read_text(encoding="utf-8")
    assert "\"split_status_current\": \"ok\"" in payload
    assert "\"go_current\": true" in payload
