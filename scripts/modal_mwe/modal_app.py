from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ads-and-modal-mwe"
FUNCTION_NAME = "remote_disambiguate"

image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    _repo_root = Path(__file__).resolve().parents[2]
    image = (
        image.pip_install_from_pyproject(str(_repo_root / "pyproject.toml"))
        .pip_install("modal")
        .add_local_python_source("author_name_disambiguation")
        .add_local_dir(
            str(_repo_root / "src" / "author_name_disambiguation" / "resources"),
            remote_path="/root/author_name_disambiguation/resources",
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


@app.function(gpu="T4", image=image, timeout=60 * 20)
def remote_disambiguate(
    *,
    publications_parquet: bytes,
    references_parquet: bytes | None = None,
    dataset_id: str,
    runtime: str = "gpu",
    infer_stage: str = "smoke",
) -> dict[str, Any]:
    from author_name_disambiguation import disambiguate_sources

    with tempfile.TemporaryDirectory(prefix="ads_and_modal_mwe_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_root = tmp_root / "inputs"
        output_root = tmp_root / "outputs"
        publications_path = _write_bytes(input_root / "publications.parquet", publications_parquet)
        references_path = _write_bytes(input_root / "references.parquet", references_parquet)

        result = disambiguate_sources(
            publications_path=publications_path,
            references_path=references_path,
            output_dir=output_root,
            runtime=runtime,
            dataset_id=dataset_id,
            infer_stage=infer_stage,
            progress=False,
        )

        summary_payload = (
            json.loads(result.summary_path.read_text(encoding="utf-8"))
            if result.summary_path is not None and result.summary_path.exists()
            else {}
        )

        return {
            "run_id": str(result.run_id),
            "dataset_id": str(dataset_id),
            "runtime": str(runtime),
            "infer_stage": str(infer_stage),
            "source_author_assignments": _read_artifact(result.source_author_assignments_path),
            "author_entities": _read_artifact(result.author_entities_path),
            "mention_clusters": _read_artifact(result.mention_clusters_path),
            "summary": json.dumps(summary_payload, ensure_ascii=False).encode("utf-8"),
        }


@app.local_entrypoint()
def main() -> None:
    print(
        "Call this app locally via `scripts/modal_mwe/client_mwe.py run ...`."
    )
