from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from author_name_disambiguation._modal_backend import ModalCostResult, resolve_modal_actual_cost
from author_name_disambiguation.common.cli_ui import CliUI, get_active_ui
from author_name_disambiguation.defaults import resolve_fixed_model_bundle_path
from author_name_disambiguation.infer_sources import InferSourcesResult, run_infer_sources
from author_name_disambiguation.progress import ProgressHandler

UserRuntime = Literal["auto", "gpu", "cpu"]
UserInferStage = Literal["full", "incremental", "smoke", "mini", "mid"]
ProgressStyle = Literal["compact", "verbose"]
UserBackend = Literal["local", "modal"]
UserModalGpu = Literal["t4", "l4"]


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


def _resolve_user_runtime(runtime: UserRuntime, *, backend: UserBackend) -> str:
    normalized = str(runtime or "auto").strip().lower() or "auto"
    if normalized == "auto":
        if backend == "modal":
            return "gpu"
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


def _resolve_user_backend(backend: UserBackend) -> UserBackend:
    normalized = str(backend or "local").strip().lower() or "local"
    if normalized not in {"local", "modal"}:
        raise ValueError(f"Unsupported backend={backend!r}. The public package supports only ['local', 'modal'].")
    return cast(UserBackend, normalized)


def disambiguate_sources(
    publications_path: str | Path,
    *,
    references_path: str | Path | None = None,
    output_dir: str | Path,
    backend: UserBackend = "local",
    runtime: UserRuntime = "auto",
    modal_gpu: UserModalGpu | None = None,
    dataset_id: str | None = None,
    infer_stage: UserInferStage = "full",
    force: bool = False,
    progress: bool = True,
    progress_style: ProgressStyle = "compact",
    progress_handler: ProgressHandler | None = None,
) -> InferSourcesResult:
    resolved_backend = _resolve_user_backend(backend)
    resolved_runtime = _resolve_user_runtime(runtime, backend=resolved_backend)
    resolved_dataset_id = _derive_dataset_id(
        publications_path=publications_path,
        output_dir=output_dir,
        dataset_id=dataset_id,
    )
    created_ui = None
    if resolved_backend != "modal" and progress_handler is None and get_active_ui() is None:
        created_ui = CliUI(total_steps=8, progress=bool(progress), progress_style=str(progress_style))
    try:
        return run_infer_sources(
            publications_path=publications_path,
            references_path=references_path,
            output_root=output_dir,
            dataset_id=resolved_dataset_id,
            model_bundle=resolve_fixed_model_bundle_path(),
            backend=resolved_backend,
            infer_stage=infer_stage,
            runtime_mode=resolved_runtime,
            modal_gpu=modal_gpu,
            force=bool(force),
            progress=bool(progress),
            progress_handler=progress_handler,
        )
    finally:
        if created_ui is not None:
            created_ui.close()


def resolve_modal_cost(output_dir: str | Path) -> ModalCostResult:
    return resolve_modal_actual_cost(output_dir=output_dir)
