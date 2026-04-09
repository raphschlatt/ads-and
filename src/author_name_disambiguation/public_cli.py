from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from author_name_disambiguation.public_api import disambiguate_sources


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
    if warnings_list:
        lines.append("Warnings: " + ", ".join(str(w) for w in warnings_list))
    return "\n".join(lines)


def cmd_infer(args):
    result = disambiguate_sources(
        publications_path=args.publications_path,
        references_path=args.references_path,
        output_dir=args.output_dir,
        runtime=args.runtime,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ads-and public inference CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("infer", help="Disambiguate one ADS dataset with the bundled baseline model.")
    sp.add_argument("--publications-path", required=True)
    sp.add_argument("--references-path", default=None)
    sp.add_argument("--output-dir", required=True)
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

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_library_noise(bool(getattr(args, "quiet_libs", False)))
    return args.func(args)


if __name__ == "__main__":
    main()
