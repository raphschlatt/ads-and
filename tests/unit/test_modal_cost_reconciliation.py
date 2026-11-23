from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from author_name_disambiguation import _modal_backend as modal_backend
from author_name_disambiguation.approaches.nand.export import build_source_author_assignments
from author_name_disambiguation.data.prepare_ads import prepare_ads_source_data
from author_name_disambiguation.defaults import resolve_fixed_model_bundle_path
from author_name_disambiguation.infer_sources import InferSourcesRequest


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
        gpu_type="L4",
        run_started_at=_utc(2026, 4, 13, 10, 5),
        run_finished_at=_utc(2026, 4, 13, 10, 32),
    )

    assert lookup["query_start_utc"] == "2026-04-13T10:00:00Z"
    assert lookup["query_end_exclusive_utc"] == "2026-04-13T11:00:00Z"
    assert lookup["exact_cost_available_after_utc"] == "2026-04-13T11:10:00Z"
    assert lookup["billing_resolution"] == "h"
    assert lookup["gpu_type"] == "L4"


def test_build_modal_lookup_cross_hour_window() -> None:
    lookup = modal_backend._build_modal_lookup(
        app_id="ap-123",
        app_name="ads-and-modal",
        gpu_type="T4",
        run_started_at=_utc(2026, 4, 13, 10, 55),
        run_finished_at=_utc(2026, 4, 13, 11, 2),
    )

    assert lookup["query_start_utc"] == "2026-04-13T10:00:00Z"
    assert lookup["query_end_exclusive_utc"] == "2026-04-13T12:00:00Z"
    assert lookup["exact_cost_available_after_utc"] == "2026-04-13T12:10:00Z"


def test_existing_modal_gpu_type_reads_summary(tmp_path) -> None:
    _write_summary(
        tmp_path,
        {
            "app_id": "ap-123",
            "app_name": "ads-and-modal",
            "gpu_type": "L4",
            "billing_resolution": "h",
            "query_start_utc": "2026-04-13T10:00:00Z",
            "query_end_exclusive_utc": "2026-04-13T11:00:00Z",
            "exact_cost_available_after_utc": "2026-04-13T11:10:00Z",
        },
    )

    assert modal_backend._existing_modal_gpu_type(tmp_path) == "L4"


def test_run_modal_infer_sources_reuses_outputs_only_for_identical_request(tmp_path, monkeypatch) -> None:
    publications_path = tmp_path / "publications.parquet"

    def _write_publications(title: str) -> None:
        pd.DataFrame(
            [{"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": title, "Abstract_en": "Abstract"}]
        ).to_parquet(publications_path, index=False)

    def _parquet_bytes(frame: pd.DataFrame) -> bytes:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue()

    class _FakeRemote:
        calls = 0

        def remote(self, *, publications_parquet: bytes, dataset_id: str, **_kwargs):
            type(self).calls += 1
            staged = tmp_path / f"staged_{self.calls}.parquet"
            staged.write_bytes(publications_parquet)
            prepared = prepare_ads_source_data(staged)
            clusters = prepared["mentions"][["mention_id"]].copy()
            clusters["author_uid_local"] = [f"blk.{idx}" for idx in range(len(clusters))]
            clusters["author_uid"] = f"{dataset_id}::" + clusters["author_uid_local"]
            assignments, entities = build_source_author_assignments(
                publications=prepared["publications"],
                references=prepared["references"],
                canonical_records=prepared["canonical_records"],
                clusters=clusters,
                uid_scope="dataset",
                uid_namespace=dataset_id,
                return_author_entities=True,
            )
            run_id = f"remote_{self.calls}"
            return {
                "run_id": run_id,
                "source_author_assignments": _parquet_bytes(assignments),
                "author_entities": _parquet_bytes(entities),
                "mention_clusters": _parquet_bytes(clusters),
                "stage_metrics": b"{}",
                "go_no_go": json.dumps({"go": True}).encode("utf-8"),
                "summary": json.dumps({"run_id": run_id, "go": True}).encode("utf-8"),
            }

    class _FakeApp:
        @contextlib.contextmanager
        def run(self):
            yield SimpleNamespace(app_id="ap-fake")

    fake_modal_app = ModuleType("author_name_disambiguation._modal_app")
    fake_modal_app.APP_NAME = "ads-and-modal"
    fake_modal_app.app = _FakeApp()
    fake_modal_app._resolve_modal_gpu_type = lambda value: str(value or "L4").upper()
    fake_modal_app.resolve_remote_disambiguate_fn = lambda _gpu_type: _FakeRemote()
    monkeypatch.setitem(sys.modules, "author_name_disambiguation._modal_app", fake_modal_app)
    monkeypatch.setattr(modal_backend, "_require_modal", lambda: object())
    monkeypatch.setattr(modal_backend, "_load_local_modal_env", lambda: None)
    monkeypatch.setattr(modal_backend, "_require_modal_credentials", lambda: None)

    request = InferSourcesRequest(
        publications_path=publications_path,
        output_root=tmp_path / "out",
        dataset_id="my_ads_2026",
        model_bundle=resolve_fixed_model_bundle_path(),
        backend="modal",
        runtime_mode="gpu",
        modal_gpu="l4",
        progress=False,
    )

    _write_publications("Paper 1")
    first = modal_backend.run_modal_infer_sources(request)
    again = modal_backend.run_modal_infer_sources(request)
    assert (first.run_id, again.run_id) == ("remote_1", "remote_1")
    assert _FakeRemote.calls == 1

    # Finished results of another request are neither returned nor overwritten, and no remote run starts.
    _write_publications("Another paper")
    with pytest.raises(FileExistsError, match="publications"):
        modal_backend.run_modal_infer_sources(request)
    assert _FakeRemote.calls == 1

    forced = modal_backend.run_modal_infer_sources(dataclasses.replace(request, force=True))
    assert forced.run_id == "remote_2"
    assert _FakeRemote.calls == 2


def test_resolve_modal_actual_cost_not_yet_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_backend, "_require_modal", lambda: object())
    monkeypatch.setattr(modal_backend, "_require_modal_credentials", lambda: None)
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
    monkeypatch.setattr(modal_backend, "_require_modal_credentials", lambda: None)
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
    monkeypatch.setattr(modal_backend, "_require_modal_credentials", lambda: None)
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


def test_resolve_modal_actual_cost_errors_when_summary_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_backend, "_require_modal", lambda: object())
    monkeypatch.setattr(modal_backend, "_require_modal_credentials", lambda: None)

    with pytest.raises(RuntimeError, match="No modal summary found"):
        modal_backend.resolve_modal_actual_cost(output_dir=tmp_path, now_utc=_utc(2026, 4, 13, 11, 15))


def test_require_modal_credentials_names_missing_env_vars(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID, MODAL_TOKEN_SECRET"):
        modal_backend._require_modal_credentials()
