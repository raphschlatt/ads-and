from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal
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


def _load_summary(payload: bytes | None) -> dict[str, Any]:
    if payload is None:
        return {}
    text = payload.decode("utf-8").strip()
    if not text:
        return {}
    return dict(json.loads(text))


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
) -> dict[str, Any]:
    localized = dict(summary)
    localized["output_root"] = str(output_dir)
    localized["summary_path"] = str(output_dir / "summary.json")
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
    localized["modal_mwe"] = {
        "app_name": APP_NAME,
        "function_name": FUNCTION_NAME,
        "transport": "modal_sdk_remote_function",
        "publications_staging_bytes": int(staging_sizes["publications"]),
        "references_staging_bytes": int(staging_sizes["references"]),
    }
    return localized


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
    publications_input = Path(publications_path).expanduser().resolve()
    references_input = None if references_path is None else Path(references_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_dataset_id = _derive_dataset_id(
        publications_path=publications_input,
        output_dir=output_root,
        dataset_id=dataset_id,
    )

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

        try:
            remote_function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
            remote_result = remote_function.remote(
                publications_parquet=publications_payload,
                references_parquet=references_payload,
                dataset_id=resolved_dataset_id,
                runtime=str(runtime),
                infer_stage=str(infer_stage),
            )
        except Exception as exc:
            raise RuntimeError(
                "Modal MWE call failed. Run `modal setup`, deploy "
                f"`scripts/modal_mwe/modal_app.py`, and ensure app={APP_NAME!r} exists."
            ) from exc

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
        _load_summary(remote_result.get("summary")),
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
    )
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return ModalMweResult(
        run_id=str(remote_result.get("run_id") or summary.get("run_id") or ""),
        output_dir=output_root,
        publications_disambiguated_path=publications_disambiguated_path,
        references_disambiguated_path=references_disambiguated_path,
        source_author_assignments_path=assignments_path,
        author_entities_path=author_entities_path,
        mention_clusters_path=mention_clusters_path,
        summary_path=summary_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone ADS-and Modal MWE.")
    parser.add_argument("--publications", required=True, help="Path to publications.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for local outputs")
    parser.add_argument("--references", default=None, help="Optional path to references.parquet")
    parser.add_argument("--dataset-id", default=None, help="Optional dataset id override")
    parser.add_argument("--runtime", default="gpu", choices=["gpu", "cpu"], help="Remote runtime mode")
    parser.add_argument(
        "--infer-stage",
        default="smoke",
        choices=["smoke", "mini", "mid", "full", "incremental"],
        help="Inference subset stage to run remotely",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = disambiguate_sources_modal_mwe(
        publications_path=args.publications,
        references_path=args.references,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        runtime=args.runtime,
        infer_stage=args.infer_stage,
    )
    print(json.dumps({"run_id": result.run_id, "summary_path": str(result.summary_path)}, indent=2))


if __name__ == "__main__":
    main()
