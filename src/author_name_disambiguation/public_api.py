from __future__ import annotations

from pathlib import Path
from typing import Literal

from author_name_disambiguation.common.cli_ui import CliUI, get_active_ui
from author_name_disambiguation.defaults import resolve_fixed_model_bundle_path
from author_name_disambiguation.infer_sources import InferSourcesResult, run_infer_sources
from author_name_disambiguation.progress import ProgressHandler

UserRuntime = Literal["auto", "gpu", "cpu"]
UserInferStage = Literal["full", "incremental", "smoke", "mini", "mid"]
ProgressStyle = Literal["compact", "verbose"]


def _derive_dataset_id(*, publications_path: str | Path, output_dir: str | Path, dataset_id: str | None) -> str:
    if dataset_id is not None and str(dataset_id).strip():
        return str(dataset_id).strip()
    output_name = Path(output_dir).expanduser().name.strip()
    if output_name not in {"", ".", ".."}:
        return output_name
    publications_name = Path(publications_path).expanduser().stem.strip()
    if publications_name:
        return publications_name
    return "dataset"


def _resolve_user_runtime(runtime: UserRuntime) -> str:
    normalized = str(runtime or "auto").strip().lower() or "auto"
    if normalized == "auto":
        try:
            import torch

            return "gpu" if bool(torch.cuda.is_available()) else "cpu"
        except Exception:
            return "cpu"
    if normalized not in {"gpu", "cpu"}:
        raise ValueError(
            f"Unsupported runtime={runtime!r}. The public package supports only ['auto', 'cpu', 'gpu']."
        )
    return normalized


def disambiguate_sources(
    publications_path: str | Path,
    *,
    references_path: str | Path | None = None,
    output_dir: str | Path,
    runtime: UserRuntime = "auto",
    dataset_id: str | None = None,
    infer_stage: UserInferStage = "full",
    force: bool = False,
    progress: bool = True,
    progress_style: ProgressStyle = "compact",
    progress_handler: ProgressHandler | None = None,
) -> InferSourcesResult:
    resolved_runtime = _resolve_user_runtime(runtime)
    resolved_dataset_id = _derive_dataset_id(
        publications_path=publications_path,
        output_dir=output_dir,
        dataset_id=dataset_id,
    )
    created_ui = None
    if progress_handler is None and get_active_ui() is None:
        created_ui = CliUI(total_steps=8, progress=bool(progress), progress_style=str(progress_style))
    try:
        return run_infer_sources(
            publications_path=publications_path,
            references_path=references_path,
            output_root=output_dir,
            dataset_id=resolved_dataset_id,
            model_bundle=resolve_fixed_model_bundle_path(),
            infer_stage=infer_stage,
            runtime_mode=resolved_runtime,
            force=bool(force),
            progress=bool(progress),
            progress_handler=progress_handler,
        )
    finally:
        if created_ui is not None:
            created_ui.close()
