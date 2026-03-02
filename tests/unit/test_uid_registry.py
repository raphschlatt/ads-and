from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.uid_registry import assign_registry_uids, load_uid_registry, save_uid_registry


def test_uid_registry_assigns_and_reuses_ids(tmp_path: Path):
    registry_path = tmp_path / "uid_registry" / "ads_prod.json"
    registry = load_uid_registry(registry_path, namespace="ads_prod")

    clusters = pd.DataFrame(
        [
            {"mention_id": "b1::0", "block_key": "j.doe", "author_uid": "j.doe::0", "author_uid_local": "j.doe::0"},
            {"mention_id": "b2::0", "block_key": "j.doe", "author_uid": "j.doe::0", "author_uid_local": "j.doe::0"},
        ]
    )
    out1, reg1, meta1 = assign_registry_uids(clusters=clusters, registry=registry, uid_namespace="ads_prod")
    save_uid_registry(registry_path, reg1)

    assert out1["author_uid"].nunique() == 1
    first_uid = out1["author_uid"].iloc[0]
    assert first_uid.startswith("ads_prod::au")
    assert meta1.clusters_new == 1
    assert meta1.clusters_reused == 0
    assert meta1.local_to_global_valid is True
    assert meta1.local_to_global_max_nunique == 1
    assert meta1.local_to_global_violations == 0

    registry_loaded = load_uid_registry(registry_path, namespace="ads_prod")
    out2, reg2, meta2 = assign_registry_uids(clusters=clusters, registry=registry_loaded, uid_namespace="ads_prod")
    assert out2["author_uid"].nunique() == 1
    assert out2["author_uid"].iloc[0] == first_uid
    assert meta2.clusters_reused == 1
    assert meta2.clusters_new == 0
    assert reg2["next_id"] == reg1["next_id"]
    assert meta2.local_to_global_valid is True
    assert meta2.local_to_global_max_nunique == 1
    assert meta2.local_to_global_violations == 0


def test_uid_registry_merges_conflicting_known_ids():
    clusters = pd.DataFrame(
        [
            {"mention_id": "b1::0", "block_key": "j.doe", "author_uid": "j.doe::0", "author_uid_local": "j.doe::0"},
            {"mention_id": "b3::0", "block_key": "j.doe", "author_uid": "j.doe::0", "author_uid_local": "j.doe::0"},
        ]
    )
    registry = {
        "schema_version": "v1",
        "uid_namespace": "ads_prod",
        "next_id": 3,
        "mention_to_uid": {
            "b1::0": "ads_prod::au000000002",
            "b3::0": "ads_prod::au000000001",
        },
        "aliases": {},
    }
    out, reg_out, meta = assign_registry_uids(clusters=clusters, registry=registry, uid_namespace="ads_prod")
    assert out["author_uid"].nunique() == 1
    assert out["author_uid"].iloc[0] == "ads_prod::au000000001"
    assert meta.clusters_merged_conflicts == 1
    assert reg_out["aliases"]["ads_prod::au000000002"] == "ads_prod::au000000001"
    assert meta.local_to_global_valid is True
    assert meta.local_to_global_max_nunique == 1
    assert meta.local_to_global_violations == 0


def test_uid_registry_interleaved_rows_keep_local_to_global_consistent():
    clusters = pd.DataFrame(
        [
            {"mention_id": "a::0", "block_key": "a", "author_uid": "a::0", "author_uid_local": "a::0"},
            {"mention_id": "b::0", "block_key": "b", "author_uid": "b::0", "author_uid_local": "b::0"},
            {"mention_id": "a::1", "block_key": "a", "author_uid": "a::0", "author_uid_local": "a::0"},
            {"mention_id": "b::1", "block_key": "b", "author_uid": "b::0", "author_uid_local": "b::0"},
        ]
    )
    registry = {
        "schema_version": "v1",
        "uid_namespace": "ads_prod",
        "next_id": 1,
        "mention_to_uid": {},
        "aliases": {},
    }
    out, _, meta = assign_registry_uids(clusters=clusters, registry=registry, uid_namespace="ads_prod")
    local_to_global = out.groupby("author_uid_local")["author_uid"].nunique()
    assert int(local_to_global.max()) == 1
    assert meta.local_to_global_valid is True
    assert meta.local_to_global_max_nunique == 1
    assert meta.local_to_global_violations == 0
