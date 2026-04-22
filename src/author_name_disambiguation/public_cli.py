from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from author_name_disambiguation.public_api import disambiguate_sources, resolve_modal_cost


def _configure_library_noise(quiet_libraries: bool) -> None:
    if not quiet_libraries:
        return

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_LOG_LEVEL"] = "3"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    warnings.filterwarnings(
        "ignore",
        message=r".*`resume_download` is deprecated.*",
        category=FutureWarning,
    )

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    try:  # pragma: no cover
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass

    try:  # pragma: no cover
        from huggingface_hub.utils import disable_progress_bars, logging as hf_logging

        disable_progress_bars()
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:  # pragma: no cover
        import absl.logging as absl_logging

        absl_logging.set_verbosity("error")
    except Exception:
        pass


def _format_human_count(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _format_human_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}s"
    except Exception:
        return str(value)


def _format_human_bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        size = float(value)
    except Exception:
        return str(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _build_infer_human_summary(summary: dict[str, Any]) -> str:
    counts = dict(summary.get("counts", {}) or {})
    stage_seconds = dict(summary.get("stage_seconds", {}) or {})
    warnings_list = list(summary.get("warnings", []) or [])
    outputs = dict(summary.get("outputs", {}) or {})
    runtime_bits = [
        f"mode={summary.get('runtime_mode', 'n/a')}",
        f"device={summary.get('resolved_device', 'n/a')}",
        f"backend={summary.get('runtime_backend', 'n/a')}",
        f"cluster={summary.get('clustering_backend', 'n/a')}",
    ]
    counts_bits = [
        f"publications={_format_human_count(counts.get('publications'))}",
        f"references={_format_human_count(counts.get('references'))}",
        f"mentions={_format_human_count(counts.get('mentions'))}",
        f"clusters={_format_human_count(counts.get('clusters'))}",
        f"authors_mapped={_format_human_count(counts.get('authors_mapped'))}/{_format_human_count(counts.get('authors_total'))}",
    ]
    stage_bits = [
        f"load={_format_human_seconds(stage_seconds.get('load_inputs'))}",
        f"preflight={_format_human_seconds(stage_seconds.get('preflight'))}",
        f"names={_format_human_seconds(stage_seconds.get('name_embeddings'))}",
        f"texts={_format_human_seconds(stage_seconds.get('text_embeddings'))}",
        f"pairs={_format_human_seconds(stage_seconds.get('pair_inference'))}",
        f"cluster={_format_human_seconds(stage_seconds.get('clustering'))}",
        f"export={_format_human_seconds(stage_seconds.get('export'))}",
        f"total={_format_human_seconds(stage_seconds.get('total'))}",
    ]
    lines = [
        "ADS inference complete",
        f"GO: {summary.get('go')} | " + " | ".join(runtime_bits),
        "Counts: " + " | ".join(counts_bits),
        "Stage times: " + " | ".join(stage_bits),
        f"Output root: {summary.get('output_root', 'n/a')}",
        f"Publications: {outputs.get('publications_disambiguated_path', 'n/a')}",
    ]
    references_path = outputs.get("references_disambiguated_path")
    if references_path:
        lines.append(f"References: {references_path}")
    lines.extend(
        [
            f"Assignments: {outputs.get('source_author_assignments_path', 'n/a')}",
            f"Summary: {summary.get('summary_path', 'n/a')}",
        ]
    )
    modal_payload = dict(summary.get("modal", {}) or {})
    if modal_payload:
        app_name = modal_payload.get("app_name") or "ads-and-modal"
        mode = modal_payload.get("mode") or "ephemeral_app_run"
        gpu_type = str(modal_payload.get("gpu_type") or "").strip()
        staging_bits = [f"pubs {_format_human_bytes(modal_payload.get('publications_staging_bytes'))}"]
        references_bytes = modal_payload.get("references_staging_bytes")
        if references_bytes:
            staging_bits.append(f"refs {_format_human_bytes(references_bytes)}")
        lines.extend(
            [
                (
                    f"Modal: app={app_name} ({mode})"
                    if not gpu_type
                    else f"Modal: app={app_name} ({mode}) | gpu={gpu_type}"
                ),
                f"       app_id={modal_payload.get('app_id', 'n/a')}",
                f"       staging={' · '.join(staging_bits)}",
                f"       billing window={modal_payload.get('query_start_utc', 'n/a')} → "
                f"{modal_payload.get('query_end_exclusive_utc', 'n/a')} (resolution {modal_payload.get('billing_resolution', 'h')})",
                f"       exact cost available after {modal_payload.get('exact_cost_available_after_utc', 'n/a')}"
                " — run `ads-and cost --output-dir <run_output_dir>`",
            ]
        )
        if modal_payload.get("actual_cost_usd") is not None:
            lines.append(f"       actual cost (USD)={modal_payload['actual_cost_usd']}")
    if warnings_list:
        lines.append("Warnings: " + ", ".join(str(w) for w in warnings_list))
    return "\n".join(lines)


def _build_cost_payload(result: Any) -> dict[str, Any]:
    payload = dataclasses.asdict(result)
    cost_report_path = payload.get("cost_report_path")
    if cost_report_path is not None:
        payload["cost_report_path"] = str(cost_report_path)
    return payload


def _build_cost_human_summary(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "n/a")
    lines = [f"Modal cost lookup: {status}"]
    lines.append(f"App ID: {payload.get('app_id', 'n/a')}")
    lines.append(f"Exact cost available after: {payload.get('exact_cost_available_after_utc', 'n/a')}")
    if payload.get("actual_cost_usd") is not None:
        lines.append(f"Actual cost (USD): {payload['actual_cost_usd']}")
    if payload.get("cost_report_path"):
        lines.append(f"Cost report: {payload['cost_report_path']}")
    if payload.get("reason"):
        lines.append(f"Reason: {payload['reason']}")
    return "\n".join(lines)


def cmd_infer(args):
    result = disambiguate_sources(
        publications_path=args.publications_path,
        references_path=args.references_path,
        output_dir=args.output_dir,
        backend=args.backend,
        runtime=args.runtime,
        modal_gpu=args.modal_gpu,
        dataset_id=args.dataset_id,
        infer_stage=args.infer_stage,
        force=bool(args.force),
        progress=bool(args.progress),
        progress_style=getattr(args, "progress_style", "compact"),
    )
    if result.summary_path is None:
        raise RuntimeError("infer completed without summary_path.")
    payload = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    if bool(getattr(args, "json_output", False)):
        print(json.dumps(payload, indent=2))
        return payload
    print(_build_infer_human_summary(payload))
    return payload


def cmd_cost(args):
    result = resolve_modal_cost(output_dir=args.output_dir)
    payload = _build_cost_payload(result)
    if bool(getattr(args, "json_output", False)):
        print(json.dumps(payload, indent=2))
        return payload
    print(_build_cost_human_summary(payload))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Disambiguate author names in ADS parquet datasets with the bundled baseline model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("infer", help="Disambiguate one ADS dataset with the bundled baseline model.")
    sp.add_argument("--publications-path", required=True)
    sp.add_argument("--references-path", default=None)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--backend", choices=["local", "modal"], default="local")
    sp.add_argument("--modal-gpu", choices=["t4", "l4"], default=None)
    sp.add_argument("--dataset-id", default=None)
    sp.add_argument("--infer-stage", choices=["smoke", "mini", "mid", "full", "incremental"], default="full")
    sp.add_argument("--runtime", choices=["auto", "gpu", "cpu"], default="auto")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--progress", dest="progress", action="store_true")
    sp.add_argument("--no-progress", dest="progress", action="store_false")
    sp.set_defaults(progress=True)
    sp.add_argument("--verbose-progress", dest="progress_style", action="store_const", const="verbose")
    sp.set_defaults(progress_style="compact")
    sp.add_argument("--quiet-libs", dest="quiet_libs", action="store_true")
    sp.add_argument("--verbose-libs", dest="quiet_libs", action="store_false")
    sp.set_defaults(quiet_libs=True)
    sp.add_argument("--json", dest="json_output", action="store_true")
    sp.set_defaults(json_output=False)
    sp.set_defaults(func=cmd_infer)

    cp = sub.add_parser("cost", help="Resolve exact Modal costs for a completed modal-backed inference run.")
    cp.add_argument("--output-dir", required=True)
    cp.add_argument("--quiet-libs", dest="quiet_libs", action="store_true")
    cp.add_argument("--verbose-libs", dest="quiet_libs", action="store_false")
    cp.set_defaults(quiet_libs=True)
    cp.add_argument("--json", dest="json_output", action="store_true")
    cp.set_defaults(json_output=False)
    cp.set_defaults(func=cmd_cost)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_library_noise(bool(getattr(args, "quiet_libs", False)))
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
