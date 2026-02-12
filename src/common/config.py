from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_project_root(start: str | Path | None = None) -> Path:
    base = Path(start) if start is not None else Path.cwd()
    for candidate in [base, *base.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate.resolve()
    return base.resolve()


def resolve_path(path_value: str | Path, project_root: str | Path | None = None) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    root = Path(project_root) if project_root is not None else find_project_root(Path.cwd())
    return (root / p).resolve()


def resolve_existing_path(
    path_value: str | Path,
    project_root: str | Path | None = None,
    extra_bases: Iterable[str | Path] | None = None,
) -> Path | None:
    p = Path(path_value)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        # First, as-is relative to current cwd.
        candidates.append((Path.cwd() / p).resolve())

        root = Path(project_root) if project_root is not None else find_project_root(Path.cwd())
        candidates.append((root / p).resolve())

        # Walk parents of cwd for notebook kernels started in subdirs.
        for parent in [Path.cwd(), *Path.cwd().parents]:
            candidates.append((parent / p).resolve())

    if extra_bases:
        for b in extra_bases:
            candidates.append((Path(b) / p).resolve())

    seen = set()
    dedup = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            dedup.append(c)

    for c in dedup:
        if c.exists():
            return c
    return None


def resolve_paths_config(paths_cfg: Dict[str, Any], project_root: str | Path | None = None) -> Dict[str, Any]:
    root = Path(project_root) if project_root is not None else find_project_root(Path.cwd())
    out = dict(paths_cfg)
    out["project_root"] = str(root)

    for section in ("data", "artifacts"):
        sec = dict(out.get(section, {}))
        for key, value in sec.items():
            if isinstance(value, str):
                sec[key] = str(resolve_path(value, project_root=root))
        out[section] = sec

    return out
