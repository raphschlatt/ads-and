from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from author_name_disambiguation.common.cache_ops import resolve_shared_cache_root


def load_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {target}")
    return dict(payload)


def build_workspace_paths(
    *,
    data_root: str | Path,
    artifacts_root: str | Path,
    raw_lspo_parquet: str | Path | None = None,
    raw_lspo_h5: str | Path | None = None,
) -> dict[str, Any]:
    data_root_path = Path(data_root).expanduser().resolve()
    artifacts_root_path = Path(artifacts_root).expanduser().resolve()

    data_cfg: dict[str, Any] = {
        "interim_dir": str(data_root_path / "interim"),
        "processed_dir": str(data_root_path / "processed"),
        "subset_cache_dir": str(data_root_path / "subsets" / "cache"),
        "subset_manifest_dir": str(data_root_path / "subsets" / "manifests"),
    }
    if raw_lspo_parquet is not None:
        data_cfg["raw_lspo_parquet"] = str(Path(raw_lspo_parquet).expanduser().resolve())
    if raw_lspo_h5 is not None:
        data_cfg["raw_lspo_h5"] = str(Path(raw_lspo_h5).expanduser().resolve())

    artifacts_cfg = {
        "root": str(artifacts_root_path),
        "embeddings_dir": str(artifacts_root_path / "embeddings"),
        "checkpoints_dir": str(artifacts_root_path / "checkpoints"),
        "pair_scores_dir": str(artifacts_root_path / "pair_scores"),
        "clusters_dir": str(artifacts_root_path / "clusters"),
        "metrics_dir": str(artifacts_root_path / "metrics"),
        "models_dir": str(artifacts_root_path / "models"),
    }
    return {"data": data_cfg, "artifacts": artifacts_cfg}


def build_run_dirs(data_cfg: dict[str, Any], artifacts_cfg: dict[str, Any], run_id: str) -> dict[str, Path]:
    shared_root = resolve_shared_cache_root(data_cfg)
    artifacts_root = artifacts_cfg.get("root")
    if artifacts_root is None:
        for key in ["metrics_dir", "checkpoints_dir", "pair_scores_dir", "clusters_dir", "embeddings_dir"]:
            value = artifacts_cfg.get(key)
            if value:
                artifacts_root = str(Path(value).expanduser().resolve().parent)
                break
    if artifacts_root is None:
        raise ValueError("artifacts_cfg must define `root` or at least one artifacts directory.")
    models_root = Path(str(artifacts_cfg.get("models_dir", Path(str(artifacts_root)) / "models")))
    return {
        "metrics": Path(artifacts_cfg["metrics_dir"]) / run_id,
        "checkpoints": Path(artifacts_cfg["checkpoints_dir"]) / run_id,
        "pair_scores": Path(artifacts_cfg["pair_scores_dir"]) / run_id,
        "clusters": Path(artifacts_cfg["clusters_dir"]) / run_id,
        "models": models_root / run_id,
        "embeddings": Path(artifacts_cfg["embeddings_dir"]) / run_id,
        "subset_cache": Path(data_cfg["subset_cache_dir"]) / run_id,
        "subset_manifests": Path(data_cfg["subset_manifest_dir"]),
        "interim": Path(data_cfg["interim_dir"]),
        "processed": Path(data_cfg["processed_dir"]),
        "shared_cache_root": shared_root,
        "shared_subsets": shared_root / "subsets",
        "shared_embeddings": shared_root / "embeddings",
        "shared_pairs": shared_root / "pairs",
        "shared_pair_scores": shared_root / "pair_scores",
        "shared_eps_sweeps": shared_root / "eps_sweeps",
    }


def write_latest_run_context(
    run_id: str,
    run_dirs: dict[str, Path],
    output_path: str | Path,
    stage: str | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "run_dirs": {key: str(value) for key, value in run_dirs.items()},
    }
    if extras:
        payload.update(extras)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def read_latest_run_context(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Latest run context not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_id(manual_run_id: str | None, latest_context_path: str | Path) -> str:
    if manual_run_id and manual_run_id.strip():
        return manual_run_id.strip()
    ctx = read_latest_run_context(latest_context_path)
    candidate = str(ctx.get("run_id") or "").strip()
    if not candidate:
        raise ValueError("run_id could not be resolved from latest run context.")
    return candidate


def write_run_consistency(
    run_id: str,
    run_stage: str,
    run_dirs: dict[str, Path],
    output_path: str | Path,
    extras: dict[str, Any] | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "run_stage": run_stage,
        "paths": {key: str(value) for key, value in run_dirs.items()},
    }
    if extras:
        payload.update(extras)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target
