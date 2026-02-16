from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import cli
from src.common.cache_ops import hash_file


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _paths_cfg(tmp_path: Path) -> Path:
    payload = {
        "project_root": str(tmp_path),
        "data": {
            "raw_lspo_parquet": str(tmp_path / "data/raw/lspo/mock.parquet"),
            "raw_lspo_h5": str(tmp_path / "data/raw/lspo/mock.h5"),
            "raw_ads_publications": str(tmp_path / "data/raw/ads/pubs.jsonl"),
            "raw_ads_references": str(tmp_path / "data/raw/ads/refs.json"),
            "interim_dir": str(tmp_path / "data/interim"),
            "processed_dir": str(tmp_path / "data/processed"),
            "subset_cache_dir": str(tmp_path / "data/subsets/cache"),
            "subset_manifest_dir": str(tmp_path / "data/subsets/manifests"),
        },
        "artifacts": {
            "root": str(tmp_path / "artifacts"),
            "embeddings_dir": str(tmp_path / "artifacts/embeddings"),
            "checkpoints_dir": str(tmp_path / "artifacts/checkpoints"),
            "pair_scores_dir": str(tmp_path / "artifacts/pair_scores"),
            "clusters_dir": str(tmp_path / "artifacts/clusters"),
            "metrics_dir": str(tmp_path / "artifacts/metrics"),
        },
    }
    return _write_yaml(tmp_path / "cfg/paths.yaml", payload)


def _mini_mentions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mention_id": "m0::0",
                "bibcode": "b0",
                "author_idx": 0,
                "author_raw": "A",
                "title": "t",
                "abstract": "a",
                "year": 2000,
                "source_type": "toy",
                "block_key": "a.block",
            }
        ]
    )


def test_cache_doctor_and_purge_stale_subsets(tmp_path: Path, capsys):
    paths_cfg = _paths_cfg(tmp_path)
    shared_dir = tmp_path / "data/cache/_shared/subsets"
    shared_dir.mkdir(parents=True, exist_ok=True)
    lspo_path = shared_dir / "lspo_mentions_smoke_seed11_target5000_cfgdeadbeef_srcabc.parquet"
    ads_path = shared_dir / "ads_mentions_smoke_seed11_target5000_cfgdeadbeef_srcabc.parquet"
    _mini_mentions().to_parquet(lspo_path, index=False)
    _mini_mentions().to_parquet(ads_path, index=False)

    parser = cli.build_parser()
    args = parser.parse_args(["cache", "doctor", "--paths-config", str(paths_cfg)])
    args.func(args)
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["counts"]["stale_subsets"] >= 1
    assert "legacy_pair_scores_detected" in doctor_payload
    assert "promotable_legacy_hits" in doctor_payload

    args = parser.parse_args(
        ["cache", "purge", "--paths-config", str(paths_cfg), "--target", "stale-subsets"]
    )
    args.func(args)
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["dry_run"] is True
    assert lspo_path.exists()

    args = parser.parse_args(
        ["cache", "purge", "--paths-config", str(paths_cfg), "--target", "stale-subsets", "--yes"]
    )
    args.func(args)
    purge_payload = json.loads(capsys.readouterr().out)
    assert purge_payload["dry_run"] is False
    assert not lspo_path.exists()
    assert not ads_path.exists()


def test_cache_doctor_and_purge_redundant_run_copies(tmp_path: Path, capsys):
    paths_cfg = _paths_cfg(tmp_path)
    shared_path = tmp_path / "data/cache/_shared/embeddings/ads_chars2vec_deadbeef.npy"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shared_path, np.ones((4, 4), dtype=np.float32))

    run_path = tmp_path / "artifacts/embeddings/smoke_run_1/ads_chars2vec_smoke.npy"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(run_path, np.load(shared_path))

    refs_path = tmp_path / "artifacts/metrics/smoke_run_1/00_cache_refs.json"
    refs_path.parent.mkdir(parents=True, exist_ok=True)
    refs_path.write_text(
        json.dumps(
            {
                "run_id": "smoke_run_1",
                "cache_refs": [
                    {
                        "artifact_type": "embedding_ads_chars",
                        "artifact_id": "deadbeef",
                        "shared_path": str(shared_path),
                        "run_path": str(run_path),
                        "materialization_mode": "copy",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    parser = cli.build_parser()
    args = parser.parse_args(["cache", "doctor", "--paths-config", str(paths_cfg)])
    args.func(args)
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["counts"]["redundant_run_copies"] >= 1

    args = parser.parse_args(
        ["cache", "purge", "--paths-config", str(paths_cfg), "--target", "redundant-run-copies", "--yes"]
    )
    args.func(args)
    purge_payload = json.loads(capsys.readouterr().out)
    assert purge_payload["candidate_count"] >= 1
    assert run_path.exists()
    samefile = False
    try:
        samefile = os.path.samefile(run_path, shared_path)
    except Exception:
        samefile = False
    assert samefile or hash_file(run_path) == hash_file(shared_path)


def test_cache_doctor_and_purge_legacy_pair_scores_unused(tmp_path: Path, capsys):
    paths_cfg = _paths_cfg(tmp_path)
    pair_scores_dir = tmp_path / "data/cache/_shared/pair_scores"
    pair_scores_dir.mkdir(parents=True, exist_ok=True)

    legacy_used = pair_scores_dir / "ads_pair_scores_deadbeef.parquet"
    legacy_unused = pair_scores_dir / "ads_pair_scores_unused.parquet"
    modern_v2 = pair_scores_dir / "ads_pair_scores_v2_abcd1234.parquet"

    sample = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "mention_id_1": "m1",
                "mention_id_2": "m2",
                "block_key": "b",
                "cosine_sim": 0.8,
                "distance": 0.2,
            }
        ]
    )
    sample.to_parquet(legacy_used, index=False)
    sample.to_parquet(legacy_unused, index=False)
    sample.to_parquet(modern_v2, index=False)

    refs_path = tmp_path / "artifacts/metrics/smoke_run_2/00_cache_refs.json"
    refs_path.parent.mkdir(parents=True, exist_ok=True)
    refs_path.write_text(
        json.dumps(
            {
                "run_id": "smoke_run_2",
                "cache_refs": [
                    {
                        "artifact_type": "pair_scores",
                        "artifact_id": "deadbeef",
                        "shared_path": str(legacy_used),
                        "run_path": str(tmp_path / "artifacts/pair_scores/smoke_run_2/ads_pair_scores_smoke.parquet"),
                        "materialization_mode": "copy",
                        "cache_schema_version": "v1",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    parser = cli.build_parser()
    args = parser.parse_args(["cache", "doctor", "--paths-config", str(paths_cfg)])
    args.func(args)
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["counts"]["legacy_pair_scores_detected"] == 2
    assert doctor_payload["counts"]["promotable_legacy_hits"] == 1

    args = parser.parse_args(
        ["cache", "purge", "--paths-config", str(paths_cfg), "--target", "legacy-pair-scores-unused", "--yes"]
    )
    args.func(args)
    purge_payload = json.loads(capsys.readouterr().out)
    assert purge_payload["candidate_count"] == 1
    assert purge_payload["purged_count"] == 1
    assert legacy_used.exists()
    assert not legacy_unused.exists()
    assert modern_v2.exists()
