from __future__ import annotations

import json
import tempfile
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path
from typing import Any

import modal

from author_name_disambiguation.defaults import resolve_fixed_model_bundle_path
from author_name_disambiguation.infer_sources import InferSourcesRequest
from author_name_disambiguation.source_inference import run_source_inference


APP_NAME = "ads-and-modal"
FUNCTION_NAME = "remote_disambiguate"
DEFAULT_GPU = "T4"
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24
PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_REMOTE_ROOT = "/root/author_name_disambiguation/resources"


def _base_runtime_requirements() -> list[str]:
    try:
        declared = list(requires("ads-and") or [])
    except PackageNotFoundError:
        return []
    return [entry for entry in declared if "extra ==" not in entry]


image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    image = (
        image.pip_install(*_base_runtime_requirements(), "modal>=1.4,<2")
        .add_local_python_source("author_name_disambiguation")
        .add_local_dir(
            str(PACKAGE_ROOT / "resources"),
            remote_path=RESOURCE_REMOTE_ROOT,
        )
    )

app = modal.App(APP_NAME)


def _write_bytes(path: Path, payload: bytes | None) -> Path | None:
    if payload is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _read_artifact(path: Path | None) -> bytes | None:
    if path is None or not path.exists():
        return None
    return path.read_bytes()


@app.function(gpu=DEFAULT_GPU, image=image, timeout=DEFAULT_TIMEOUT_SECONDS)
def remote_disambiguate(
    *,
    publications_parquet: bytes,
    references_parquet: bytes | None = None,
    dataset_id: str,
    runtime_mode: str = "gpu",
    infer_stage: str = "full",
    uid_scope: str = "dataset",
    uid_namespace: str | None = None,
    device: str = "auto",
    precision_mode: str = "fp32",
    specter_runtime_backend: str | None = None,
    cluster_backend: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ads_and_modal_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_root = tmp_root / "inputs"
        output_root = tmp_root / "outputs"
        publications_path = _write_bytes(input_root / "publications.parquet", publications_parquet)
        references_path = _write_bytes(input_root / "references.parquet", references_parquet)

        result = run_source_inference(
            InferSourcesRequest(
                publications_path=publications_path,
                references_path=references_path,
                output_root=output_root,
                dataset_id=str(dataset_id),
                model_bundle=resolve_fixed_model_bundle_path(),
                uid_scope=str(uid_scope),
                uid_namespace=None if uid_namespace is None else str(uid_namespace),
                infer_stage=str(infer_stage),
                runtime_mode=str(runtime_mode),
                device=str(device),
                precision_mode=str(precision_mode),
                specter_runtime_backend=None if specter_runtime_backend is None else str(specter_runtime_backend),
                cluster_backend=None if cluster_backend is None else str(cluster_backend),
                force=bool(force),
                progress=False,
            )
        )

        summary_payload = (
            json.loads(result.summary_path.read_text(encoding="utf-8"))
            if result.summary_path is not None and result.summary_path.exists()
            else {}
        )

        return {
            "run_id": str(result.run_id),
            "dataset_id": str(dataset_id),
            "runtime_mode": str(runtime_mode),
            "infer_stage": str(infer_stage),
            "source_author_assignments": _read_artifact(result.source_author_assignments_path),
            "author_entities": _read_artifact(result.author_entities_path),
            "mention_clusters": _read_artifact(result.mention_clusters_path),
            "stage_metrics": _read_artifact(result.stage_metrics_path),
            "go_no_go": _read_artifact(result.go_no_go_path),
            "summary": json.dumps(summary_payload, ensure_ascii=False).encode("utf-8"),
        }
