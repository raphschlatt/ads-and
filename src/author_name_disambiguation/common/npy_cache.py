from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def fingerprint_texts(texts: Iterable[str], **settings: Any) -> str:
    """Checksum of the texts a cache is built from, in order, plus the settings that shape it."""
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
    for text in texts:
        digest.update(b"\x1f")
        digest.update(str(text).encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _fingerprint_path(target: Path) -> Path:
    return target.with_suffix(".fingerprint.json")


def _read_fingerprint(target: Path) -> str | None:
    try:
        return json.loads(_fingerprint_path(target).read_text(encoding="utf-8")).get("fingerprint")
    except Exception:
        return None


def atomic_save_npy(path: str | Path, arr: np.ndarray, *, fingerprint: str | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # An old fingerprint must never vouch for new content.
    _fingerprint_path(target).unlink(missing_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".npy", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, np.asarray(arr))
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    if fingerprint is not None:
        _fingerprint_path(target).write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    return target


def load_validated_npy(
    path: str | Path,
    *,
    validator: Callable[[np.ndarray], bool] | None = None,
    description: str = "NumPy cache",
    fingerprint: str | None = None,
) -> np.ndarray | None:
    """Load a cached array, or return None when it must be recomputed.

    With ``fingerprint``, the cache counts only if it was saved with the same fingerprint.
    """
    target = Path(path)
    if not target.exists():
        return None
    if fingerprint is not None and _read_fingerprint(target) != fingerprint:
        return None
    try:
        arr = np.load(target, allow_pickle=False)
    except Exception as exc:
        warnings.warn(
            f"{description} at {target} is unreadable ({exc!r}); recomputing.",
            RuntimeWarning,
        )
        return None
    if validator is not None:
        try:
            is_valid = bool(validator(arr))
        except Exception as exc:
            warnings.warn(
                f"{description} at {target} could not be validated ({exc!r}); recomputing.",
                RuntimeWarning,
            )
            return None
        if not is_valid:
            warnings.warn(
                f"{description} at {target} failed validation for shape={tuple(arr.shape)!r}; recomputing.",
                RuntimeWarning,
            )
            return None
    return np.asarray(arr)
