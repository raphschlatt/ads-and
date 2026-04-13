from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

from author_name_disambiguation import _modal_backend as modal_backend


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _write_summary(tmp_path, modal_payload: dict) -> None:
    payload = {
        "run_id": "infer_sources_test",
        "summary_path": str(tmp_path / "summary.json"),
        "modal": dict(modal_payload),
    }
    (tmp_path / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_modal_lookup_same_hour_window() -> None:
    lookup = modal_backend._build_modal_lookup(
        app_id="ap-123",
        app_name="ads-and-modal",
        run_started_at=_utc(2026, 4, 13, 10, 5),
        run_finished_at=_utc(2026, 4, 13, 10, 32),
    )

    assert lookup["query_start_utc"] == "2026-04-13T10:00:00Z"
    assert lookup["query_end_exclusive_utc"] == "2026-04-13T11:00:00Z"
    assert lookup["exact_cost_available_after_utc"] == "2026-04-13T11:10:00Z"
    assert lookup["billing_resolution"] == "h"


def test_build_modal_lookup_cross_hour_window() -> None:
    lookup = modal_backend._build_modal_lookup(
        app_id="ap-123",
        app_name="ads-and-modal",
        run_started_at=_utc(2026, 4, 13, 10, 55),
        run_finished_at=_utc(2026, 4, 13, 11, 2),
    )

    assert lookup["query_start_utc"] == "2026-04-13T10:00:00Z"
    assert lookup["query_end_exclusive_utc"] == "2026-04-13T12:00:00Z"
    assert lookup["exact_cost_available_after_utc"] == "2026-04-13T12:10:00Z"


def test_resolve_modal_actual_cost_not_yet_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_backend, "_require_modal", lambda: object())
    _write_summary(
        tmp_path,
        {
            "app_id": "ap-123",
            "app_name": "ads-and-modal",
            "billing_resolution": "h",
            "query_start_utc": "2026-04-13T10:00:00Z",
            "query_end_exclusive_utc": "2026-04-13T11:00:00Z",
            "exact_cost_available_after_utc": "2026-04-13T11:10:00Z",
        },
    )

    result = modal_backend.resolve_modal_actual_cost(
        output_dir=tmp_path,
        now_utc=_utc(2026, 4, 13, 11, 5),
    )

    assert result.status == "not_yet_available"
    assert result.actual_cost_usd is None
    assert result.cost_report_path is None
    assert "2026-04-13T11:10:00Z" in str(result.reason)


def test_resolve_modal_actual_cost_complete(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_backend, "_require_modal", lambda: object())
    _write_summary(
        tmp_path,
        {
            "app_id": "ap-123",
            "app_name": "ads-and-modal",
            "billing_resolution": "h",
            "query_start_utc": "2026-04-13T10:00:00Z",
            "query_end_exclusive_utc": "2026-04-13T11:00:00Z",
            "exact_cost_available_after_utc": "2026-04-13T11:10:00Z",
        },
    )

    def fake_report(*, start, end, resolution):
        assert start == _utc(2026, 4, 13, 10, 0)
        assert end == _utc(2026, 4, 13, 11, 0)
        assert resolution == "h"
        return [
            {
                "object_id": "ap-123",
                "description": "ads-and-modal",
                "environment_name": "main",
                "interval_start": _utc(2026, 4, 13, 10, 0),
                "cost": Decimal("0.1234"),
                "tags": {},
            },
            {
                "object_id": "ap-other",
                "description": "other",
                "environment_name": "main",
                "interval_start": _utc(2026, 4, 13, 10, 0),
                "cost": Decimal("9.99"),
                "tags": {},
            },
        ]

    monkeypatch.setattr(modal_backend, "_workspace_billing_report", fake_report)

    result = modal_backend.resolve_modal_actual_cost(
        output_dir=tmp_path,
        now_utc=_utc(2026, 4, 13, 11, 15),
    )

    assert result.status == "complete"
    assert result.actual_cost_usd == 0.1234
    assert result.cost_report_path == tmp_path / "modal_cost_report.json"

    report = json.loads((tmp_path / "modal_cost_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["actual_cost_usd"] == 0.1234
    assert len(report["interval_rows"]) == 1
    assert report["interval_rows"][0]["object_id"] == "ap-123"

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["modal"]["actual_cost_usd"] == 0.1234
    assert summary["modal"]["cost_report_path"] == str(tmp_path / "modal_cost_report.json")


def test_resolve_modal_actual_cost_unsupported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_backend, "_require_modal", lambda: object())
    _write_summary(
        tmp_path,
        {
            "app_id": "ap-123",
            "app_name": "ads-and-modal",
            "billing_resolution": "h",
            "query_start_utc": "2026-04-13T10:00:00Z",
            "query_end_exclusive_utc": "2026-04-13T11:00:00Z",
            "exact_cost_available_after_utc": "2026-04-13T11:10:00Z",
        },
    )

    def fake_report(*, start, end, resolution):
        raise RuntimeError("Workspace billing reports are only available on Team and Enterprise plans.")

    monkeypatch.setattr(modal_backend, "_workspace_billing_report", fake_report)

    result = modal_backend.resolve_modal_actual_cost(
        output_dir=tmp_path,
        now_utc=_utc(2026, 4, 13, 11, 15),
    )

    assert result.status == "unsupported"
    assert result.actual_cost_usd is None
    assert result.cost_report_path is None
    assert "Team and Enterprise" in str(result.reason)
