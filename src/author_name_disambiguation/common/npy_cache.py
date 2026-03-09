from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Callable

import numpy as np


def atomic_save_npy(path: str | Path, arr: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    return target


def load_validated_npy(
    path: str | Path,
    *,
    validator: Callable[[np.ndarray], bool] | None = None,
    description: str = "NumPy cache",
) -> np.ndarray | None:
    target = Path(path)
    if not target.exists():
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
