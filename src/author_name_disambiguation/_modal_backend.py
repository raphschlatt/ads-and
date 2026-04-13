from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from author_name_disambiguation.approaches.nand.export import export_source_mirrored_outputs
from author_name_disambiguation.common.io_schema import save_parquet
from author_name_disambiguation.common.pipeline_reports import load_json
from author_name_disambiguation.data.prepare_ads import _read_ads_parquet_minimal
from author_name_disambiguation.defaults import resolve_fixed_model_bundle_path
from author_name_disambiguation.source_inference import _required_outputs_exist


BILLING_RESOLUTION = "h"
BILLING_BUFFER_MINUTES = 10
COST_REPORT_FILENAME = "modal_cost_report.json"
SUMMARY_FILENAME = "summary.json"
STAGE_METRICS_FILENAME = "05_stage_metrics_infer_sources.json"
GO_NO_GO_FILENAME = "05_go_no_go_infer_sources.json"


@dataclass(slots=True)
class ModalCostResult:
    status: str
    app_id: str
    exact_cost_available_after_utc: str
    actual_cost_usd: float | None
    cost_report_path: Path | None
    reason: str | None = None


def _require_modal():
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError(
            "Modal support requires the optional dependency. Install it with "
            "`uv sync --extra modal` or `uv pip install \"ads-and[modal]\"`."
        ) from exc
    return modal


def _load_local_modal_env(env_path: str | Path | None = None) -> None:
    candidate = Path(env_path or ".env").expanduser().resolve()
    if not candidate.exists():
        return
    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        key = name.strip()
        if key not in {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}:
            continue
        cleaned = value.strip().strip("'").strip('"')
        if cleaned and key not in os.environ:
            os.environ[key] = cleaned


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _floor_to_hour(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _ceil_to_hour(value: datetime) -> datetime:
    normalized = value.astimezone(timezone.utc)
    floored = _floor_to_hour(normalized)
    if normalized == floored:
        return floored
    return floored + timedelta(hours=1)


def _build_modal_lookup(
    *,
    app_id: str,
    app_name: str,
    gpu_type: str,
    run_started_at: datetime,
    run_finished_at: datetime,
) -> dict[str, Any]:
    query_start = _floor_to_hour(run_started_at)
    query_end_exclusive = _ceil_to_hour(run_finished_at)
    exact_cost_available_after = query_end_exclusive + timedelta(minutes=BILLING_BUFFER_MINUTES)
    return {
        "app_id": str(app_id),
        "app_name": str(app_name),
        "gpu_type": str(gpu_type),
        "mode": "ephemeral_app_run",
        "transport": "modal_sdk_app_run",
        "run_started_at_utc": _format_utc(run_started_at),
        "run_finished_at_utc": _format_utc(run_finished_at),
        "billing_resolution": BILLING_RESOLUTION,
        "query_start_utc": _format_utc(query_start),
        "query_end_exclusive_utc": _format_utc(query_end_exclusive),
        "exact_cost_available_after_utc": _format_utc(exact_cost_available_after),
        "actual_cost_usd": None,
        "cost_report_path": None,
    }


def _stage_projected_parquet_bytes(input_path: str | Path, staging_path: Path) -> bytes:
    projected = _read_ads_parquet_minimal(Path(input_path))
    save_parquet(projected, staging_path, index=False)
    return staging_path.read_bytes()


def _write_bytes(path: Path, payload: bytes | None) -> Path | None:
    if payload is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _load_summary_bytes(payload: bytes | None) -> dict[str, Any]:
    if payload is None:
        return {}
    text = payload.decode("utf-8").strip()
    if not text:
        return {}
    return dict(json.loads(text))


def _build_result_paths(*, output_root: Path, references_present: bool) -> dict[str, Path | None]:
    return {
        "publications_disambiguated": output_root / "publications_disambiguated.parquet",
        "references_disambiguated": None if not references_present else output_root / "references_disambiguated.parquet",
        "source_author_assignments": output_root / "source_author_assignments.parquet",
        "author_entities": output_root / "author_entities.parquet",
        "mention_clusters": output_root / "mention_clusters.parquet",
        "stage_metrics": output_root / STAGE_METRICS_FILENAME,
        "go_no_go": output_root / GO_NO_GO_FILENAME,
    }


def _existing_result(request, *, result_paths: dict[str, Path | None]):
    from author_name_disambiguation.infer_sources import InferSourcesResult

    go_payload = load_json(result_paths["go_no_go"])
    summary_path = Path(request.output_root).expanduser().resolve() / SUMMARY_FILENAME
    summary_payload = (
        load_json(summary_path)
        if summary_path.exists()
        else {
            "dataset_id": str(request.dataset_id),
            "go": go_payload.get("go"),
            "summary_path": str(summary_path),
        }
    )
    return InferSourcesResult(
        run_id=str(summary_payload.get("run_id") or ""),
        go=go_payload.get("go"),
        output_root=Path(request.output_root).expanduser().resolve(),
        publications_disambiguated_path=result_paths["publications_disambiguated"],
        references_disambiguated_path=result_paths["references_disambiguated"],
        source_author_assignments_path=result_paths["source_author_assignments"],
        author_entities_path=result_paths["author_entities"],
        mention_clusters_path=result_paths["mention_clusters"],
        stage_metrics_path=result_paths["stage_metrics"],
        go_no_go_path=result_paths["go_no_go"],
        summary_path=summary_path,
    )


def _serialize_billing_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for row in rows:
        interval_start = row.get("interval_start")
        serialized.append(
            {
                "object_id": str(row.get("object_id") or ""),
                "description": str(row.get("description") or ""),
                "environment_name": str(row.get("environment_name") or ""),
                "interval_start_utc": (
                    None
                    if not isinstance(interval_start, datetime)
                    else _format_utc(interval_start.astimezone(timezone.utc))
                ),
                "cost_usd": float(row.get("cost") or 0.0),
                "tags": dict(row.get("tags") or {}),
            }
        )
    return serialized


def _workspace_billing_report(*, start: datetime, end: datetime, resolution: str) -> list[dict[str, Any]]:
    modal = _require_modal()
    return list(modal.billing.workspace_billing_report(start=start, end=end, resolution=resolution))


def _is_workspace_billing_unsupported(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    if not text:
        return False
    markers = (
        "team",
        "enterprise",
        "billing report",
        "billing reports",
        "not enabled",
        "not available",
        "permission denied",
        "forbidden",
        "not authorized",
        "unauthorized",
    )
    return any(marker in text for marker in markers)


def _localize_summary(
    summary: dict[str, Any],
    *,
    output_dir: Path,
    publications_output_path: Path,
    references_output_path: Path | None,
    assignments_path: Path,
    author_entities_path: Path,
    mention_clusters_path: Path,
    stage_metrics_path: Path,
    go_no_go_path: Path,
    staging_sizes: dict[str, int],
    modal_lookup: dict[str, Any],
) -> dict[str, Any]:
    localized = dict(summary)
    localized["output_root"] = str(output_dir)
    localized["summary_path"] = str(output_dir / SUMMARY_FILENAME)
    localized["publications_disambiguated_path"] = str(publications_output_path)
    localized["references_disambiguated_path"] = (
        None if references_output_path is None else str(references_output_path)
    )
    localized["source_author_assignments_path"] = str(assignments_path)
    localized["author_entities_path"] = str(author_entities_path)
    localized["mention_clusters_path"] = str(mention_clusters_path)
    localized["stage_metrics_path"] = str(stage_metrics_path)
    localized["go_no_go_path"] = str(go_no_go_path)
    localized["outputs"] = {
        "publications_disambiguated_path": str(publications_output_path),
        "references_disambiguated_path": None if references_output_path is None else str(references_output_path),
        "source_author_assignments_path": str(assignments_path),
        "author_entities_path": str(author_entities_path),
        "mention_clusters_path": str(mention_clusters_path),
        "stage_metrics_path": str(stage_metrics_path),
        "go_no_go_path": str(go_no_go_path),
    }
    localized["modal"] = {
        **dict(modal_lookup),
        "publications_staging_bytes": int(staging_sizes["publications"]),
        "references_staging_bytes": int(staging_sizes["references"]),
    }
    return localized


def _resolve_modal_progress(request) -> bool:
    caller_progress = bool(getattr(request, "progress", True))
    caller_handler = getattr(request, "progress_handler", None)
    if caller_handler is not None:
        warnings.warn(
            "progress_handler is ignored on the modal backend; the remote container cannot "
            "stream structured progress events to a client-side handler. Use backend='local' "
            "for custom progress handlers.",
            RuntimeWarning,
            stacklevel=3,
        )
    return caller_progress and caller_handler is None


def _existing_modal_gpu_type(output_root: Path) -> str | None:
    summary_path = output_root / SUMMARY_FILENAME
    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
    except Exception:
        return None
    modal_meta = dict(summary.get("modal") or {})
    value = str(modal_meta.get("gpu_type") or "").strip().upper()
    return value or None


def _validate_modal_request(request) -> None:
    publications_path = Path(request.publications_path).expanduser()
    references_path = None if request.references_path is None else Path(request.references_path).expanduser()
    if publications_path.suffix.lower() != ".parquet":
        raise ValueError("backend='modal' currently supports only parquet publications input.")
    if references_path is not None and references_path.suffix.lower() != ".parquet":
        raise ValueError("backend='modal' currently supports only parquet references input.")
    if request.scratch_dir is not None:
        raise ValueError("backend='modal' does not support scratch_dir overrides.")
    if request.cluster_config is not None or request.gates_config is not None:
        raise ValueError("backend='modal' does not support cluster_config or gates_config overrides in v1.")
    bundled_path = resolve_fixed_model_bundle_path().resolve()
    if request.model_bundle is not None and Path(request.model_bundle).expanduser().resolve() != bundled_path:
        raise ValueError("backend='modal' currently supports only the packaged fixed model bundle.")


def run_modal_infer_sources(request):
    from author_name_disambiguation._modal_app import APP_NAME
    from author_name_disambiguation._modal_app import _resolve_modal_gpu_type
    from author_name_disambiguation._modal_app import app as modal_app
    from author_name_disambiguation._modal_app import resolve_remote_disambiguate_fn
    from author_name_disambiguation.infer_sources import InferSourcesResult

    modal = _require_modal()
    _load_local_modal_env()
    _validate_modal_request(request)

    output_root = Path(request.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    references_present = request.references_path is not None
    result_paths = _build_result_paths(output_root=output_root, references_present=references_present)
    stale_cost_report = output_root / COST_REPORT_FILENAME
    requested_gpu_type = _resolve_modal_gpu_type(getattr(request, "modal_gpu", None))

    remote_progress = _resolve_modal_progress(request)

    if not request.force and _required_outputs_exist(output_root, references_present=references_present):
        existing_gpu_type = _existing_modal_gpu_type(output_root)
        if existing_gpu_type == requested_gpu_type:
            if remote_progress:
                print(
                    f"ADS · Modal · reused existing infer outputs · gpu={requested_gpu_type}",
                    file=sys.stderr,
                    flush=True,
                )
            return _existing_result(request, result_paths=result_paths)
        if remote_progress:
            print(
                "ADS · Modal · existing outputs skipped due to gpu mismatch "
                f"(requested={requested_gpu_type} existing={existing_gpu_type or 'unknown'})",
                file=sys.stderr,
                flush=True,
            )

    if stale_cost_report.exists():
        stale_cost_report.unlink()

    with tempfile.TemporaryDirectory(prefix="ads_and_modal_client_") as tmp_dir:
        staging_root = Path(tmp_dir)
        publications_payload = _stage_projected_parquet_bytes(
            request.publications_path,
            staging_root / "publications.minimal.parquet",
        )
        references_payload = (
            None
            if request.references_path is None
            else _stage_projected_parquet_bytes(
                request.references_path,
                staging_root / "references.minimal.parquet",
            )
        )

        if remote_progress:
            print(
                f"ADS · Modal · dataset={request.dataset_id} · infer_stage={request.infer_stage} · output_root={output_root}",
                file=sys.stderr,
                flush=True,
            )

        run_started_at = _utc_now()
        remote_result: dict[str, Any]
        app_id = ""
        remote_disambiguate = resolve_remote_disambiguate_fn(requested_gpu_type)
        output_cm = modal.enable_output() if remote_progress else contextlib.nullcontext()
        with output_cm, modal_app.run() as running_app:
            app_id = str(getattr(running_app, "app_id", "") or "")
            remote_result = remote_disambiguate.remote(
                publications_parquet=publications_payload,
                references_parquet=references_payload,
                dataset_id=str(request.dataset_id),
                runtime_mode=str(request.runtime_mode or "gpu"),
                infer_stage=str(request.infer_stage),
                uid_scope=str(request.uid_scope),
                uid_namespace=None if request.uid_namespace is None else str(request.uid_namespace),
                device=str(request.device),
                precision_mode=str(request.precision_mode),
                specter_runtime_backend=(
                    None if request.specter_runtime_backend is None else str(request.specter_runtime_backend)
                ),
                cluster_backend=None if request.cluster_backend is None else str(request.cluster_backend),
                force=bool(request.force),
                progress=remote_progress,
            )
            if not app_id:
                app_id = str(getattr(running_app, "app_id", "") or "")
        run_finished_at = _utc_now()

    assignments_path = _write_bytes(result_paths["source_author_assignments"], remote_result.get("source_author_assignments"))
    author_entities_path = _write_bytes(result_paths["author_entities"], remote_result.get("author_entities"))
    mention_clusters_path = _write_bytes(result_paths["mention_clusters"], remote_result.get("mention_clusters"))
    stage_metrics_path = _write_bytes(result_paths["stage_metrics"], remote_result.get("stage_metrics"))
    go_no_go_path = _write_bytes(result_paths["go_no_go"], remote_result.get("go_no_go"))
    if None in {assignments_path, author_entities_path, mention_clusters_path, stage_metrics_path, go_no_go_path}:
        raise RuntimeError("Modal backend returned incomplete artifacts.")

    assignments = pd.read_parquet(assignments_path)
    publications_disambiguated_path = result_paths["publications_disambiguated"]
    references_disambiguated_path = result_paths["references_disambiguated"]

    export_source_mirrored_outputs(
        assignments=assignments,
        publications_path=request.publications_path,
        references_path=request.references_path,
        publications_output_path=publications_disambiguated_path,
        references_output_path=references_disambiguated_path,
    )

    modal_lookup = _build_modal_lookup(
        app_id=app_id,
        app_name=APP_NAME,
        gpu_type=requested_gpu_type,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
    )
    summary = _localize_summary(
        _load_summary_bytes(remote_result.get("summary")),
        output_dir=output_root,
        publications_output_path=publications_disambiguated_path,
        references_output_path=references_disambiguated_path,
        assignments_path=assignments_path,
        author_entities_path=author_entities_path,
        mention_clusters_path=mention_clusters_path,
        stage_metrics_path=stage_metrics_path,
        go_no_go_path=go_no_go_path,
        staging_sizes={
            "publications": len(publications_payload),
            "references": 0 if references_payload is None else len(references_payload),
        },
        modal_lookup=modal_lookup,
    )
    summary_path = _write_json(output_root / SUMMARY_FILENAME, summary)

    return InferSourcesResult(
        run_id=str(summary.get("run_id") or remote_result.get("run_id") or ""),
        go=summary.get("go"),
        output_root=output_root,
        publications_disambiguated_path=publications_disambiguated_path,
        references_disambiguated_path=references_disambiguated_path,
        source_author_assignments_path=assignments_path,
        author_entities_path=author_entities_path,
        mention_clusters_path=mention_clusters_path,
        stage_metrics_path=stage_metrics_path,
        go_no_go_path=go_no_go_path,
        summary_path=summary_path,
    )


def resolve_modal_actual_cost(*, output_dir: str | Path, now_utc: datetime | None = None) -> ModalCostResult:
    _require_modal()
    _load_local_modal_env()
    summary_path = Path(output_dir).expanduser().resolve() / SUMMARY_FILENAME
    summary = _load_json(summary_path)
    modal_meta = dict(summary.get("modal") or {})
    app_id = str(modal_meta.get("app_id") or "").strip()
    exact_cost_available_after_utc = str(modal_meta.get("exact_cost_available_after_utc") or "").strip()
    if not app_id or not exact_cost_available_after_utc:
        raise RuntimeError(f"Missing modal lookup metadata in {summary_path}. Run `ads-and infer --backend modal` first.")

    now_value = _utc_now() if now_utc is None else now_utc.astimezone(timezone.utc)
    available_after = _parse_utc(exact_cost_available_after_utc)
    if now_value < available_after:
        return ModalCostResult(
            status="not_yet_available",
            app_id=app_id,
            exact_cost_available_after_utc=exact_cost_available_after_utc,
            actual_cost_usd=None,
            cost_report_path=None,
            reason=f"Exact cost will be queryable after {exact_cost_available_after_utc}.",
        )

    query_start = _parse_utc(str(modal_meta.get("query_start_utc") or ""))
    query_end = _parse_utc(str(modal_meta.get("query_end_exclusive_utc") or ""))
    try:
        rows = _workspace_billing_report(
            start=query_start,
            end=query_end,
            resolution=str(modal_meta.get("billing_resolution") or BILLING_RESOLUTION),
        )
    except Exception as exc:
        if _is_workspace_billing_unsupported(exc):
            return ModalCostResult(
                status="unsupported",
                app_id=app_id,
                exact_cost_available_after_utc=exact_cost_available_after_utc,
                actual_cost_usd=None,
                cost_report_path=None,
                reason=str(exc).strip() or exc.__class__.__name__,
            )
        raise RuntimeError("Modal billing lookup failed.") from exc

    matched_rows = [row for row in rows if str(row.get("object_id") or "") == app_id]
    if not matched_rows:
        return ModalCostResult(
            status="not_yet_available",
            app_id=app_id,
            exact_cost_available_after_utc=exact_cost_available_after_utc,
            actual_cost_usd=None,
            cost_report_path=None,
            reason=f"No billing rows for app_id={app_id} were available yet.",
        )

    total_cost = sum((Decimal(row.get("cost") or 0) for row in matched_rows), Decimal("0"))
    cost_report = {
        "status": "complete",
        "app_id": app_id,
        "app_name": str(modal_meta.get("app_name") or ""),
        "billing_resolution": str(modal_meta.get("billing_resolution") or BILLING_RESOLUTION),
        "query_start_utc": _format_utc(query_start),
        "query_end_exclusive_utc": _format_utc(query_end),
        "resolved_at_utc": _format_utc(now_value),
        "actual_cost_usd": float(total_cost),
        "interval_rows": _serialize_billing_rows(matched_rows),
    }
    report_path = _write_json(Path(output_dir).expanduser().resolve() / COST_REPORT_FILENAME, cost_report)

    modal_meta["actual_cost_usd"] = float(total_cost)
    modal_meta["cost_report_path"] = str(report_path)
    summary["modal"] = modal_meta
    _write_json(summary_path, summary)

    return ModalCostResult(
        status="complete",
        app_id=app_id,
        exact_cost_available_after_utc=exact_cost_available_after_utc,
        actual_cost_usd=float(total_cost),
        cost_report_path=report_path,
    )
