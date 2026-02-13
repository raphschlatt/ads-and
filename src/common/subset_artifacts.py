from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.common.io_schema import read_parquet, save_parquet


@dataclass(frozen=True)
class SubsetIdentity:
    stage: str
    seed: int
    target_mentions: int | None
    target_tag: str
    source_fp: str
    cfg_fp: str
    sampler_version: str
    subset_tag: str


@dataclass(frozen=True)
class SubsetPaths:
    shared_dir: Path
    lspo_shared: Path
    ads_shared: Path
    lspo_legacy: Path
    ads_legacy: Path


@dataclass(frozen=True)
class ManifestPaths:
    lspo_primary: Path
    ads_primary: Path
    lspo_legacy: Path
    ads_legacy: Path


@dataclass(frozen=True)
class LoadMeta:
    source: str
    identity: SubsetIdentity
    lspo_path: Path
    ads_path: Path


def _file_stamp(path: Path) -> str:
    st = Path(path).stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def _normalize_subset_sampling(run_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = run_cfg.get("subset_sampling") or {}
    if not isinstance(raw, Mapping):
        return {}
    normalized: dict[str, Any] = {}
    if "target_mean_block_size" in raw and raw.get("target_mean_block_size") is not None:
        normalized["target_mean_block_size"] = float(raw["target_mean_block_size"])
    return normalized


def compute_source_fp(lspo_interim_path: Path, ads_interim_path: Path) -> str:
    payload = f"lspo:{_file_stamp(lspo_interim_path)}|ads:{_file_stamp(ads_interim_path)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def compute_subset_identity(
    run_cfg: Mapping[str, Any],
    source_fp: str,
    sampler_version: str = "v2",
) -> SubsetIdentity:
    stage = str(run_cfg["stage"])
    seed = int(run_cfg.get("seed", 11))
    raw_target = run_cfg.get("subset_target_mentions")
    target_mentions = None if raw_target is None else int(raw_target)
    target_tag = "full" if target_mentions is None else str(target_mentions)
    subset_sampling = _normalize_subset_sampling(run_cfg)

    cfg_payload = {
        "sampler_version": str(sampler_version),
        "stage": stage,
        "seed": seed,
        "target_tag": target_tag,
        "subset_sampling": subset_sampling,
    }
    cfg_blob = json.dumps(cfg_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    cfg_fp = hashlib.sha1(cfg_blob.encode("utf-8")).hexdigest()[:8]
    subset_tag = f"{stage}_seed{seed}_target{target_tag}_cfg{cfg_fp}_src{source_fp}"

    return SubsetIdentity(
        stage=stage,
        seed=seed,
        target_mentions=target_mentions,
        target_tag=target_tag,
        source_fp=source_fp,
        cfg_fp=cfg_fp,
        sampler_version=str(sampler_version),
        subset_tag=subset_tag,
    )


def resolve_shared_subset_paths(
    data_cfg: Mapping[str, Any],
    identity: SubsetIdentity,
) -> SubsetPaths:
    shared_dir = Path(data_cfg["subset_cache_dir"]) / "_shared"
    lspo_shared = shared_dir / f"lspo_mentions_{identity.subset_tag}.parquet"
    ads_shared = shared_dir / f"ads_mentions_{identity.subset_tag}.parquet"
    # Legacy stage-local naming.
    lspo_legacy = Path(data_cfg["subset_cache_dir"]) / f"lspo_mentions_{identity.stage}.parquet"
    ads_legacy = Path(data_cfg["subset_cache_dir"]) / f"ads_mentions_{identity.stage}.parquet"
    return SubsetPaths(
        shared_dir=shared_dir,
        lspo_shared=lspo_shared,
        ads_shared=ads_shared,
        lspo_legacy=lspo_legacy,
        ads_legacy=ads_legacy,
    )


def resolve_manifest_paths(
    run_id: str,
    manifest_dir: Path,
    identity: SubsetIdentity,
    run_stage: str,
) -> ManifestPaths:
    manifest_dir = Path(manifest_dir)
    return ManifestPaths(
        lspo_primary=manifest_dir / f"{run_id}_lspo_{identity.subset_tag}_manifest.parquet",
        ads_primary=manifest_dir / f"{run_id}_ads_{identity.subset_tag}_manifest.parquet",
        lspo_legacy=manifest_dir / f"{run_id}_lspo_{run_stage}_manifest.parquet",
        ads_legacy=manifest_dir / f"{run_id}_ads_{run_stage}_manifest.parquet",
    )


def resolve_shared_and_legacy_subset_paths(
    data_cfg: Mapping[str, Any],
    run_subset_dir: Path,
    identity: SubsetIdentity,
    run_stage: str,
) -> SubsetPaths:
    shared_dir = Path(data_cfg["subset_cache_dir"]) / "_shared"
    return SubsetPaths(
        shared_dir=shared_dir,
        lspo_shared=shared_dir / f"lspo_mentions_{identity.subset_tag}.parquet",
        ads_shared=shared_dir / f"ads_mentions_{identity.subset_tag}.parquet",
        lspo_legacy=Path(run_subset_dir) / f"lspo_mentions_{run_stage}.parquet",
        ads_legacy=Path(run_subset_dir) / f"ads_mentions_{run_stage}.parquet",
    )


def load_subset_mentions(
    *,
    data_cfg: Mapping[str, Any],
    run_dirs: Mapping[str, Path],
    run_cfg: Mapping[str, Any],
    run_stage: str,
    allow_legacy: bool = True,
    sampler_version: str = "v2",
) -> tuple[pd.DataFrame, pd.DataFrame, LoadMeta]:
    source_fp = compute_source_fp(
        lspo_interim_path=Path(run_dirs["interim"]) / "lspo_mentions.parquet",
        ads_interim_path=Path(run_dirs["interim"]) / "ads_mentions.parquet",
    )
    identity = compute_subset_identity(run_cfg=run_cfg, source_fp=source_fp, sampler_version=sampler_version)
    paths = resolve_shared_and_legacy_subset_paths(
        data_cfg=data_cfg,
        run_subset_dir=Path(run_dirs["subset_cache"]),
        identity=identity,
        run_stage=run_stage,
    )

    if paths.lspo_shared.exists() and paths.ads_shared.exists():
        return (
            read_parquet(paths.lspo_shared),
            read_parquet(paths.ads_shared),
            LoadMeta(source="shared", identity=identity, lspo_path=paths.lspo_shared, ads_path=paths.ads_shared),
        )

    if allow_legacy and paths.lspo_legacy.exists() and paths.ads_legacy.exists():
        return (
            read_parquet(paths.lspo_legacy),
            read_parquet(paths.ads_legacy),
            LoadMeta(source="legacy", identity=identity, lspo_path=paths.lspo_legacy, ads_path=paths.ads_legacy),
        )

    raise FileNotFoundError(
        "Subset mentions not found for current config. "
        f"Checked shared: {paths.lspo_shared}, {paths.ads_shared}; "
        f"legacy: {paths.lspo_legacy}, {paths.ads_legacy}"
    )


def atomic_save_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.parent / f"{final_path.name}.tmp-{os.getpid()}"
    if tmp_path.exists():
        tmp_path.unlink()
    save_parquet(df, tmp_path, index=index)
    tmp_path.replace(final_path)
    return final_path
