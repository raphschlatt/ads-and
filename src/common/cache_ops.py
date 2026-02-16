from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


def resolve_shared_cache_root(data_cfg: Mapping[str, Any]) -> Path:
    explicit = data_cfg.get("shared_cache_dir")
    if explicit:
        return Path(str(explicit)) / "_shared"
    processed_dir = data_cfg.get("processed_dir")
    if processed_dir:
        return Path(str(processed_dir)).parent / "cache" / "_shared"
    subset_cache_dir = Path(str(data_cfg["subset_cache_dir"]))
    return subset_cache_dir.parent.parent / "cache" / "_shared"


def stable_hash(payload: Any, length: int = 12) -> str:
    if isinstance(payload, bytes):
        blob = payload
    elif isinstance(payload, str):
        blob = payload.encode("utf-8")
    else:
        blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[: int(length)]


def hash_file(path: str | Path, length: int = 12) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[: int(length)]


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise IsADirectoryError(f"Expected file path, got directory: {path}")


def link_or_copy(shared_path: str | Path, run_path: str | Path) -> str:
    src = Path(shared_path)
    dst = Path(run_path)
    if not src.exists():
        raise FileNotFoundError(f"Shared source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if dst.exists() and os.path.samefile(src, dst):
            return "existing"
    except Exception:
        pass

    if dst.exists() or dst.is_symlink():
        _remove_existing(dst)

    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        pass

    try:
        dst.symlink_to(src)
        return "symlink"
    except Exception:
        pass

    shutil.copy2(src, dst)
    return "copy"
