import json
from pathlib import Path

import pytest

from src.common.config import (
    read_latest_run_context,
    resolve_run_id,
    write_latest_run_context,
)


def test_write_and_read_latest_run_context(tmp_path: Path):
    out = tmp_path / "latest_run.json"
    run_dirs = {"metrics": tmp_path / "metrics"}

    write_latest_run_context(
        run_id="smoke_123",
        run_dirs=run_dirs,
        output_path=out,
        stage="smoke",
        extras={"x": 1},
    )

    payload = read_latest_run_context(out)
    assert payload["run_id"] == "smoke_123"
    assert payload["stage"] == "smoke"
    assert payload["x"] == 1
    assert payload["run_dirs"]["metrics"].endswith("metrics")


def test_resolve_run_id_uses_manual_first(tmp_path: Path):
    out = tmp_path / "latest_run.json"
    out.write_text(json.dumps({"run_id": "smoke_ctx"}), encoding="utf-8")

    resolved = resolve_run_id("smoke_manual", out)
    assert resolved == "smoke_manual"


def test_resolve_run_id_from_latest_context(tmp_path: Path):
    out = tmp_path / "latest_run.json"
    out.write_text(json.dumps({"run_id": "smoke_ctx"}), encoding="utf-8")

    resolved = resolve_run_id(None, out)
    assert resolved == "smoke_ctx"


def test_resolve_run_id_rejects_placeholder(tmp_path: Path):
    out = tmp_path / "latest_run.json"
    out.write_text(json.dumps({"run_id": "replace_with_run_id_from_00"}), encoding="utf-8")

    with pytest.raises(ValueError, match="placeholder"):
        resolve_run_id(None, out, allow_placeholder=False)
