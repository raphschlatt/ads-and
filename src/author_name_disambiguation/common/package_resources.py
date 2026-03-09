from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def resource_path(relative_path: str) -> Path:
    base = resources.files("author_name_disambiguation")
    target = base.joinpath(relative_path)
    with resources.as_file(target) as resolved:
        return Path(resolved)


def load_yaml_resource(relative_path: str) -> dict[str, Any]:
    with resource_path(relative_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Packaged YAML resource must contain a mapping: {relative_path}")
    return dict(payload)


def load_yaml_like(
    value: str | Path | Mapping[str, Any] | None,
    *,
    default_resource: str,
    param_name: str,
) -> dict[str, Any]:
    if value is None:
        return load_yaml_resource(default_resource)
    if isinstance(value, Mapping):
        return dict(value)

    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{param_name} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{param_name} must contain a YAML mapping: {path}")
    return dict(payload)
