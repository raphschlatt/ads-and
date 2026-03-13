from pathlib import Path
import os

import pandas as pd

from author_name_disambiguation.common.cache_ops import resolve_shared_cache_root
from author_name_disambiguation.common.io_schema import save_parquet
from author_name_disambiguation.common.subset_artifacts import (
    LSPO_SOURCE_FP_SCHEME,
    LSPO_SOURCE_FP_SCHEME_LEGACY,
    compute_ads_source_fp,
    compute_lspo_source_fp,
    compute_lspo_source_fp_legacy,
    compute_source_fp,
    compute_subset_identity,
    load_subset_mentions,
    resolve_manifest_paths,
)


def _mention_df(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mention_id": f"{prefix}::0",
                "bibcode": f"{prefix}",
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


def test_source_fingerprint_is_deterministic_and_input_sensitive(tmp_path: Path):
    lspo = tmp_path / "lspo.parquet"
    ads = tmp_path / "ads.parquet"
    save_parquet(_mention_df("lspo"), lspo, index=False)
    save_parquet(_mention_df("ads"), ads, index=False)

    fp1 = compute_source_fp(lspo, ads)
    fp2 = compute_source_fp(lspo, ads)
    assert fp1 == fp2

    save_parquet(pd.concat([_mention_df("ads"), _mention_df("ads2")], ignore_index=True), ads, index=False)
    fp3 = compute_source_fp(lspo, ads)
    assert fp3 != fp1


def test_lspo_ads_source_fingerprints_are_separate(tmp_path: Path):
    lspo = tmp_path / "lspo.parquet"
    ads = tmp_path / "ads.parquet"
    save_parquet(_mention_df("lspo"), lspo, index=False)
    save_parquet(_mention_df("ads"), ads, index=False)

    lspo_fp_1 = compute_lspo_source_fp(lspo)
    ads_fp_1 = compute_ads_source_fp(ads)
    assert lspo_fp_1 != ads_fp_1

    save_parquet(pd.concat([_mention_df("lspo"), _mention_df("lspo2")], ignore_index=True), lspo, index=False)
    lspo_fp_2 = compute_lspo_source_fp(lspo)
    ads_fp_2 = compute_ads_source_fp(ads)
    assert lspo_fp_2 != lspo_fp_1
    assert ads_fp_2 == ads_fp_1


def test_lspo_source_fingerprint_ignores_file_touch(tmp_path: Path):
    lspo = tmp_path / "lspo.parquet"
    save_parquet(_mention_df("lspo"), lspo, index=False)

    stable_fp_1 = compute_lspo_source_fp(lspo)
    legacy_fp_1 = compute_lspo_source_fp_legacy(lspo)
    st = lspo.stat()
    os.utime(lspo, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    stable_fp_2 = compute_lspo_source_fp(lspo)
    legacy_fp_2 = compute_lspo_source_fp_legacy(lspo)

    assert LSPO_SOURCE_FP_SCHEME == "prepared_mentions_content_v1"
    assert LSPO_SOURCE_FP_SCHEME_LEGACY == "file_stamp_v1"
    assert stable_fp_1 == stable_fp_2
    assert legacy_fp_1 != legacy_fp_2


def test_subset_identity_changes_when_sampling_config_changes():
    base_cfg = {"stage": "smoke", "seed": 11, "subset_target_mentions": 5000, "subset_sampling": {"target_mean_block_size": 4}}
    id1 = compute_subset_identity(base_cfg, source_fp="abc123def456", sampler_version="v3")

    changed_cfg = {"stage": "smoke", "seed": 11, "subset_target_mentions": 5000, "subset_sampling": {"target_mean_block_size": 5}}
    id2 = compute_subset_identity(changed_cfg, source_fp="abc123def456", sampler_version="v3")

    assert id1.subset_tag != id2.subset_tag
    assert id1.cfg_fp != id2.cfg_fp


def test_load_subset_mentions_prefers_shared_over_legacy(tmp_path: Path):
    data_cfg = {"subset_cache_dir": str(tmp_path / "subsets" / "cache")}
    run_dirs = {
        "interim": tmp_path / "interim",
        "subset_cache": tmp_path / "subsets" / "cache" / "run1",
    }
    for p in run_dirs.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    run_cfg = {"stage": "smoke", "seed": 11, "subset_target_mentions": 5000, "subset_sampling": {"target_mean_block_size": 4}}
    run_stage = "smoke"

    save_parquet(_mention_df("lspo_src"), Path(run_dirs["interim"]) / "lspo_mentions.parquet", index=False)
    save_parquet(_mention_df("ads_src"), Path(run_dirs["interim"]) / "ads_mentions.parquet", index=False)

    source_fp = compute_source_fp(
        Path(run_dirs["interim"]) / "lspo_mentions.parquet",
        Path(run_dirs["interim"]) / "ads_mentions.parquet",
    )
    identity = compute_subset_identity(run_cfg, source_fp=source_fp, sampler_version="v3")

    shared_dir = resolve_shared_cache_root(data_cfg) / "subsets"
    shared_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(_mention_df("lspo_shared"), shared_dir / f"lspo_mentions_{identity.subset_tag}.parquet", index=False)
    save_parquet(_mention_df("ads_shared"), shared_dir / f"ads_mentions_{identity.subset_tag}.parquet", index=False)

    save_parquet(_mention_df("lspo_legacy"), Path(run_dirs["subset_cache"]) / "lspo_mentions_smoke.parquet", index=False)
    save_parquet(_mention_df("ads_legacy"), Path(run_dirs["subset_cache"]) / "ads_mentions_smoke.parquet", index=False)

    lspo, ads, meta = load_subset_mentions(
        data_cfg=data_cfg,
        run_dirs=run_dirs,
        run_cfg=run_cfg,
        run_stage=run_stage,
        allow_legacy=True,
        sampler_version="v3",
    )
    assert meta.source == "shared"
    assert lspo.iloc[0]["bibcode"] == "lspo_shared"
    assert ads.iloc[0]["bibcode"] == "ads_shared"


def test_load_subset_mentions_falls_back_to_legacy(tmp_path: Path):
    data_cfg = {"subset_cache_dir": str(tmp_path / "subsets" / "cache")}
    run_dirs = {
        "interim": tmp_path / "interim",
        "subset_cache": tmp_path / "subsets" / "cache" / "run2",
    }
    for p in run_dirs.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    run_cfg = {"stage": "smoke", "seed": 11, "subset_target_mentions": 5000, "subset_sampling": {"target_mean_block_size": 4}}

    save_parquet(_mention_df("lspo_src"), Path(run_dirs["interim"]) / "lspo_mentions.parquet", index=False)
    save_parquet(_mention_df("ads_src"), Path(run_dirs["interim"]) / "ads_mentions.parquet", index=False)
    save_parquet(_mention_df("lspo_legacy"), Path(run_dirs["subset_cache"]) / "lspo_mentions_smoke.parquet", index=False)
    save_parquet(_mention_df("ads_legacy"), Path(run_dirs["subset_cache"]) / "ads_mentions_smoke.parquet", index=False)

    lspo, ads, meta = load_subset_mentions(
        data_cfg=data_cfg,
        run_dirs=run_dirs,
        run_cfg=run_cfg,
        run_stage="smoke",
        allow_legacy=True,
        sampler_version="v3",
    )
    assert meta.source == "legacy"
    assert lspo.iloc[0]["bibcode"] == "lspo_legacy"
    assert ads.iloc[0]["bibcode"] == "ads_legacy"


def test_manifest_paths_include_new_and_legacy_names():
    identity = compute_subset_identity(
        {"stage": "mini", "seed": 11, "subset_target_mentions": 10000, "subset_sampling": {"target_mean_block_size": 4}},
        source_fp="0123456789ab",
        sampler_version="v3",
    )
    paths = resolve_manifest_paths(
        run_id="mini_20260213T000000Z_abcd",
        manifest_dir=Path("data/subsets/manifests"),
        identity=identity,
        run_stage="mini",
    )

    assert f"_lspo_{identity.subset_tag}_manifest.parquet" in str(paths.lspo_primary)
    assert str(paths.lspo_legacy).endswith("_lspo_mini_manifest.parquet")
