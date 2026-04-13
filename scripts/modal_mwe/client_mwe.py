from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from author_name_disambiguation.approaches.nand.export import export_source_mirrored_outputs
from author_name_disambiguation.common.io_schema import save_parquet
from author_name_disambiguation.data.prepare_ads import _read_ads_parquet_minimal


APP_NAME = "ads-and-modal-mwe"
FUNCTION_NAME = "remote_disambiguate"
BILLING_RESOLUTION = "h"
BILLING_BUFFER_MINUTES = 10
COST_REPORT_FILENAME = "modal_cost_report.json"
SUMMARY_FILENAME = "summary.json"


@dataclass(slots=True)
class ModalMweResult:
    run_id: str
    output_dir: Path
    publications_disambiguated_path: Path
    references_disambiguated_path: Path | None
    source_author_assignments_path: Path
    author_entities_path: Path
    mention_clusters_path: Path
    summary_path: Path
    app_id: str
    exact_cost_available_after_utc: str


@dataclass(slots=True)
class ModalMweCostResult:
    status: str
    app_id: str
    exact_cost_available_after_utc: str
    actual_cost_usd: float | None
    cost_report_path: Path | None
    reason: str | None = None


def _derive_dataset_id(*, publications_path: Path, output_dir: Path, dataset_id: str | None) -> str:
    if dataset_id is not None and str(dataset_id).strip():
        return str(dataset_id).strip()
    output_name = output_dir.name.strip()
    if output_name not in {"", ".", ".."}:
        return output_name
    publications_name = publications_path.stem.strip()
    if publications_name:
        return publications_name
    return "dataset"


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


def _build_modal_lookup(*, app_id: str, app_name: str, run_started_at: datetime, run_finished_at: datetime) -> dict[str, Any]:
    query_start = _floor_to_hour(run_started_at)
    query_end_exclusive = _ceil_to_hour(run_finished_at)
    exact_cost_available_after = query_end_exclusive + timedelta(minutes=BILLING_BUFFER_MINUTES)
    return {
        "app_id": str(app_id),
        "app_name": str(app_name),
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


def _load_local_modal_env(env_path: str | Path | None = None) -> None:
    candidate = Path(env_path or REPO_ROOT / ".env").expanduser().resolve()
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


def _load_summary_bytes(payload: bytes | None) -> dict[str, Any]:
    if payload is None:
        return {}
    text = payload.decode("utf-8").strip()
    if not text:
        return {}
    return dict(json.loads(text))


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_required_artifacts(output_root: Path, remote_result: dict[str, Any]) -> tuple[Path, Path, Path]:
    assignments_path = _write_bytes(
        output_root / "source_author_assignments.parquet",
        remote_result.get("source_author_assignments"),
    )
    author_entities_path = _write_bytes(
        output_root / "author_entities.parquet",
        remote_result.get("author_entities"),
    )
    mention_clusters_path = _write_bytes(
        output_root / "mention_clusters.parquet",
        remote_result.get("mention_clusters"),
    )
    if assignments_path is None or author_entities_path is None or mention_clusters_path is None:
        raise RuntimeError("Modal MWE returned incomplete core artifacts.")
    return assignments_path, author_entities_path, mention_clusters_path


def _localize_summary(
    summary: dict[str, Any],
    *,
    output_dir: Path,
    publications_output_path: Path,
    references_output_path: Path | None,
    assignments_path: Path,
    author_entities_path: Path,
    mention_clusters_path: Path,
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
    localized["stage_metrics_path"] = None
    localized["go_no_go_path"] = None
    localized["outputs"] = {
        "publications_disambiguated_path": str(publications_output_path),
        "references_disambiguated_path": None if references_output_path is None else str(references_output_path),
        "source_author_assignments_path": str(assignments_path),
        "author_entities_path": str(author_entities_path),
        "mention_clusters_path": str(mention_clusters_path),
        "stage_metrics_path": None,
        "go_no_go_path": None,
    }
    localized["modal"] = {
        **dict(modal_lookup),
        "function_name": FUNCTION_NAME,
        "publications_staging_bytes": int(staging_sizes["publications"]),
        "references_staging_bytes": int(staging_sizes["references"]),
    }
    return localized


def _import_modal_app_module():
    try:
        from scripts.modal_mwe import modal_app as modal_app_module
    except ImportError:
        import modal_app as modal_app_module

    return modal_app_module


def _workspace_billing_report(*, start: datetime, end: datetime, resolution: str) -> list[dict[str, Any]]:
    import modal.billing

    return list(
        modal.billing.workspace_billing_report(
            start=start,
            end=end,
            resolution=resolution,
        )
    )


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


def _resolve_summary_path(*, output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / SUMMARY_FILENAME


def disambiguate_sources_modal_mwe(
    *,
    publications_path: str | Path,
    output_dir: str | Path,
    references_path: str | Path | None = None,
    dataset_id: str | None = None,
    runtime: str = "gpu",
    infer_stage: str = "smoke",
) -> ModalMweResult:
    _load_local_modal_env()
    modal_app_module = _import_modal_app_module()

    publications_input = Path(publications_path).expanduser().resolve()
    references_input = None if references_path is None else Path(references_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stale_cost_report = output_root / COST_REPORT_FILENAME
    if stale_cost_report.exists():
        stale_cost_report.unlink()
    resolved_dataset_id = _derive_dataset_id(
        publications_path=publications_input,
        output_dir=output_root,
        dataset_id=dataset_id,
    )

    remote_result: dict[str, Any] | None = None
    run_started_at: datetime | None = None
    app_id = ""

    try:
        with tempfile.TemporaryDirectory(prefix="ads_and_modal_mwe_client_") as tmp_dir:
            staging_root = Path(tmp_dir)
            publications_payload = _stage_projected_parquet_bytes(
                publications_input,
                staging_root / "publications.minimal.parquet",
            )
            references_payload = (
                None
                if references_input is None
                else _stage_projected_parquet_bytes(
                    references_input,
                    staging_root / "references.minimal.parquet",
                )
            )

            run_started_at = _utc_now()
            with modal_app_module.app.run() as running_app:
                app_id = str(getattr(running_app, "app_id", "") or "")
                remote_result = modal_app_module.remote_disambiguate.remote(
                    publications_parquet=publications_payload,
                    references_parquet=references_payload,
                    dataset_id=resolved_dataset_id,
                    runtime=str(runtime),
                    infer_stage=str(infer_stage),
                )
                if not app_id:
                    app_id = str(getattr(running_app, "app_id", "") or "")
    except Exception as exc:
        raise RuntimeError("Modal MWE run failed. Ensure Modal auth is configured in `.env` or via `modal setup`.") from exc

    run_finished_at = _utc_now()
    if remote_result is None:
        raise RuntimeError("Modal MWE run ended without a remote result.")
    if run_started_at is None:
        raise RuntimeError("Modal MWE run ended without a recorded Modal app start time.")

    modal_lookup = _build_modal_lookup(
        app_id=app_id,
        app_name=APP_NAME,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
    )

    assignments_path, author_entities_path, mention_clusters_path = _write_required_artifacts(
        output_root,
        remote_result,
    )

    assignments = pd.read_parquet(assignments_path)
    publications_disambiguated_path = output_root / "publications_disambiguated.parquet"
    references_disambiguated_path = (
        None if references_input is None else output_root / "references_disambiguated.parquet"
    )
    export_source_mirrored_outputs(
        assignments=assignments,
        publications_path=publications_input,
        references_path=references_input,
        publications_output_path=publications_disambiguated_path,
        references_output_path=references_disambiguated_path,
    )

    summary = _localize_summary(
        _load_summary_bytes(remote_result.get("summary")),
        output_dir=output_root,
        publications_output_path=publications_disambiguated_path,
        references_output_path=references_disambiguated_path,
        assignments_path=assignments_path,
        author_entities_path=author_entities_path,
        mention_clusters_path=mention_clusters_path,
        staging_sizes={
            "publications": len(publications_payload),
            "references": 0 if references_payload is None else len(references_payload),
        },
        modal_lookup=modal_lookup,
    )
    summary_path = _write_json(output_root / SUMMARY_FILENAME, summary)

    return ModalMweResult(
        run_id=str(remote_result.get("run_id") or summary.get("run_id") or ""),
        output_dir=output_root,
        publications_disambiguated_path=publications_disambiguated_path,
        references_disambiguated_path=references_disambiguated_path,
        source_author_assignments_path=assignments_path,
        author_entities_path=author_entities_path,
        mention_clusters_path=mention_clusters_path,
        summary_path=summary_path,
        app_id=str(modal_lookup["app_id"]),
        exact_cost_available_after_utc=str(modal_lookup["exact_cost_available_after_utc"]),
    )


def resolve_modal_actual_cost(*, output_dir: str | Path, now_utc: datetime | None = None) -> ModalMweCostResult:
    _load_local_modal_env()
    summary_path = _resolve_summary_path(output_dir=output_dir)
    summary = _load_json(summary_path)
    modal_meta = dict(summary.get("modal") or {})
    app_id = str(modal_meta.get("app_id") or "").strip()
    exact_cost_available_after_utc = str(modal_meta.get("exact_cost_available_after_utc") or "").strip()
    if not app_id or not exact_cost_available_after_utc:
        raise RuntimeError(f"Missing modal lookup metadata in {summary_path}. Run the Modal MWE first.")

    now_value = _utc_now() if now_utc is None else now_utc.astimezone(timezone.utc)
    available_after = _parse_utc(exact_cost_available_after_utc)
    if now_value < available_after:
        return ModalMweCostResult(
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
            return ModalMweCostResult(
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
        return ModalMweCostResult(
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
        "app_name": str(modal_meta.get("app_name") or APP_NAME),
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

    return ModalMweCostResult(
        status="complete",
        app_id=app_id,
        exact_cost_available_after_utc=exact_cost_available_after_utc,
        actual_cost_usd=float(total_cost),
        cost_report_path=report_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone ADS-and Modal MWE.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the standalone Modal MWE.")
    run_parser.add_argument("--publications", required=True, help="Path to publications.parquet")
    run_parser.add_argument("--output-dir", required=True, help="Directory for local outputs")
    run_parser.add_argument("--references", default=None, help="Optional path to references.parquet")
    run_parser.add_argument("--dataset-id", default=None, help="Optional dataset id override")
    run_parser.add_argument("--runtime", default="gpu", choices=["gpu", "cpu"], help="Remote runtime mode")
    run_parser.add_argument(
        "--infer-stage",
        default="smoke",
        choices=["smoke", "mini", "mid", "full", "incremental"],
        help="Inference subset stage to run remotely",
    )

    cost_parser = subparsers.add_parser("cost", help="Resolve exact Modal cost for a finished MWE run.")
    cost_parser.add_argument("--output-dir", required=True, help="Directory containing summary.json")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        result = disambiguate_sources_modal_mwe(
            publications_path=args.publications,
            references_path=args.references,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            runtime=args.runtime,
            infer_stage=args.infer_stage,
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "summary_path": str(result.summary_path),
                    "app_id": result.app_id,
                    "exact_cost_available_after_utc": result.exact_cost_available_after_utc,
                },
                indent=2,
            )
        )
        return

    if args.command == "cost":
        result = resolve_modal_actual_cost(output_dir=args.output_dir)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "app_id": result.app_id,
                    "exact_cost_available_after_utc": result.exact_cost_available_after_utc,
                    "actual_cost_usd": result.actual_cost_usd,
                    "cost_report_path": None if result.cost_report_path is None else str(result.cost_report_path),
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return

    raise RuntimeError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    main()
