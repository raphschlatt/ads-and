from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from author_name_disambiguation.defaults import resolve_fixed_model_bundle_path
from author_name_disambiguation.progress import ProgressHandler
from author_name_disambiguation.source_inference import run_source_inference


Backend = Literal["local", "modal"]
UidScope = Literal["dataset", "local", "registry"]
InferStage = Literal["smoke", "mini", "mid", "full", "incremental"]
PrecisionMode = Literal["fp32", "amp_bf16"]
ClusterBackend = Literal["auto", "sklearn_cpu", "cuml_gpu"]
SpecterRuntimeBackend = Literal["transformers", "onnx_fp32"]
RuntimeMode = Literal["gpu", "cpu"]
ModalGpu = Literal["t4", "l4"]


@dataclass(slots=True)
class InferSourcesRequest:
    publications_path: str | Path
    output_root: str | Path
    dataset_id: str
    model_bundle: str | Path | None = None
    backend: Backend = "local"
    references_path: str | Path | None = None
    scratch_dir: str | Path | None = None
    uid_scope: UidScope = "dataset"
    uid_namespace: str | None = None
    infer_stage: InferStage = "full"
    cluster_config: str | Path | Mapping[str, Any] | None = None
    gates_config: str | Path | Mapping[str, Any] | None = None
    runtime_mode: RuntimeMode | None = None
    device: str = "auto"
    precision_mode: PrecisionMode = "fp32"
    specter_runtime_backend: SpecterRuntimeBackend | None = None
    cluster_backend: ClusterBackend | None = None
    modal_gpu: ModalGpu | None = None
    force: bool = False
    progress: bool = True
    progress_handler: ProgressHandler | None = None


@dataclass(slots=True)
class InferSourcesResult:
    run_id: str
    go: bool | None
    output_root: Path
    publications_disambiguated_path: Path
    references_disambiguated_path: Path | None
    source_author_assignments_path: Path
    author_entities_path: Path
    mention_clusters_path: Path
    stage_metrics_path: Path
    go_no_go_path: Path
    summary_path: Path | None = None


def _resolve_backend(value: str | None) -> Backend:
    normalized = str(value or "local").strip().lower() or "local"
    if normalized not in {"local", "modal"}:
        raise ValueError(f"Unsupported backend={value!r}. Expected one of ['local', 'modal'].")
    return normalized


def _resolve_runtime_mode(
    *,
    backend: Backend,
    runtime_mode: RuntimeMode | None,
    specter_runtime_backend: SpecterRuntimeBackend | None,
    device: str,
) -> RuntimeMode | None:
    if runtime_mode is not None:
        return runtime_mode
    if backend == "modal":
        return "gpu"
    return None


def _resolve_modal_gpu(value: str | None) -> ModalGpu | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in {"t4", "l4"}:
        raise ValueError(f"Unsupported modal_gpu={value!r}. Expected one of ['l4', 't4'].")
    return normalized


def _run_modal_infer_sources(request: InferSourcesRequest) -> InferSourcesResult:
    from author_name_disambiguation._modal_backend import run_modal_infer_sources

    return run_modal_infer_sources(request)


def _build_infer_request(
    request: InferSourcesRequest | None = None,
    **kwargs: Any,
) -> InferSourcesRequest:
    if request is not None and kwargs:
        raise TypeError("run_infer_sources accepts either an InferSourcesRequest or keyword arguments, not both.")
    if request is None:
        if not kwargs:
            raise TypeError("run_infer_sources requires an InferSourcesRequest or keyword arguments.")
        kwargs["backend"] = _resolve_backend(kwargs.get("backend"))
        kwargs["runtime_mode"] = _resolve_runtime_mode(
            backend=kwargs["backend"],
            runtime_mode=kwargs.get("runtime_mode"),
            specter_runtime_backend=kwargs.get("specter_runtime_backend"),
            device=str(kwargs.get("device", "auto")),
        )
        kwargs["modal_gpu"] = _resolve_modal_gpu(kwargs.get("modal_gpu"))
        request = InferSourcesRequest(**kwargs)
    if not isinstance(request, InferSourcesRequest):
        raise TypeError("run_infer_sources expected an InferSourcesRequest instance.")
    resolved_backend = _resolve_backend(request.backend)
    resolved_runtime_mode = _resolve_runtime_mode(
        backend=resolved_backend,
        runtime_mode=request.runtime_mode,
        specter_runtime_backend=request.specter_runtime_backend,
        device=request.device,
    )
    resolved_modal_gpu = _resolve_modal_gpu(request.modal_gpu)
    if request.model_bundle is None:
        request = InferSourcesRequest(
            publications_path=request.publications_path,
            output_root=request.output_root,
            dataset_id=request.dataset_id,
            model_bundle=resolve_fixed_model_bundle_path(),
            backend=resolved_backend,
            references_path=request.references_path,
            scratch_dir=request.scratch_dir,
            uid_scope=request.uid_scope,
            uid_namespace=request.uid_namespace,
            infer_stage=request.infer_stage,
            cluster_config=request.cluster_config,
            gates_config=request.gates_config,
            runtime_mode=resolved_runtime_mode,
            device=request.device,
            precision_mode=request.precision_mode,
            specter_runtime_backend=request.specter_runtime_backend,
            cluster_backend=request.cluster_backend,
            modal_gpu=resolved_modal_gpu,
            force=request.force,
            progress=request.progress,
            progress_handler=request.progress_handler,
        )
    elif (
        resolved_backend != request.backend
        or resolved_runtime_mode != request.runtime_mode
        or resolved_modal_gpu != request.modal_gpu
    ):
        request = InferSourcesRequest(
            publications_path=request.publications_path,
            output_root=request.output_root,
            dataset_id=request.dataset_id,
            model_bundle=request.model_bundle,
            backend=resolved_backend,
            references_path=request.references_path,
            scratch_dir=request.scratch_dir,
            uid_scope=request.uid_scope,
            uid_namespace=request.uid_namespace,
            infer_stage=request.infer_stage,
            cluster_config=request.cluster_config,
            gates_config=request.gates_config,
            runtime_mode=resolved_runtime_mode,
            device=request.device,
            precision_mode=request.precision_mode,
            specter_runtime_backend=request.specter_runtime_backend,
            cluster_backend=request.cluster_backend,
            modal_gpu=resolved_modal_gpu,
            force=request.force,
            progress=request.progress,
            progress_handler=request.progress_handler,
        )
    return request


def run_infer_sources(request: InferSourcesRequest | None = None, **kwargs: Any) -> InferSourcesResult:
    resolved_request = _build_infer_request(request, **kwargs)
    if resolved_request.backend == "modal":
        return _run_modal_infer_sources(resolved_request)
    return run_source_inference(resolved_request)
