from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from author_name_disambiguation.defaults import resolve_fixed_model_bundle_path
from author_name_disambiguation.progress import ProgressHandler
from author_name_disambiguation.source_inference import run_source_inference


UidScope = Literal["dataset", "local", "registry"]
InferStage = Literal["smoke", "mini", "mid", "full", "incremental"]
PrecisionMode = Literal["fp32", "amp_bf16"]
ClusterBackend = Literal["auto", "sklearn_cpu", "cuml_gpu"]
SpecterRuntimeBackend = Literal["transformers", "onnx_fp32"]
RuntimeMode = Literal["gpu", "cpu", "hf"]


@dataclass(slots=True)
class InferSourcesRequest:
    publications_path: str | Path
    output_root: str | Path
    dataset_id: str
    model_bundle: str | Path | None = None
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


def _build_infer_request(
    request: InferSourcesRequest | None = None,
    **kwargs: Any,
) -> InferSourcesRequest:
    if request is not None and kwargs:
        raise TypeError("run_infer_sources accepts either an InferSourcesRequest or keyword arguments, not both.")
    if request is None:
        if not kwargs:
            raise TypeError("run_infer_sources requires an InferSourcesRequest or keyword arguments.")
        if "runtime_mode" not in kwargs and "specter_runtime_backend" not in kwargs and str(kwargs.get("device", "auto")).strip().lower() == "auto":
            try:
                import torch

                kwargs["runtime_mode"] = "gpu" if bool(torch.cuda.is_available()) else "cpu"
            except Exception:
                kwargs["runtime_mode"] = "cpu"
        request = InferSourcesRequest(**kwargs)
    if not isinstance(request, InferSourcesRequest):
        raise TypeError("run_infer_sources expected an InferSourcesRequest instance.")
    if request.model_bundle is None:
        request = InferSourcesRequest(
            publications_path=request.publications_path,
            output_root=request.output_root,
            dataset_id=request.dataset_id,
            model_bundle=resolve_fixed_model_bundle_path(),
            references_path=request.references_path,
            scratch_dir=request.scratch_dir,
            uid_scope=request.uid_scope,
            uid_namespace=request.uid_namespace,
            infer_stage=request.infer_stage,
            cluster_config=request.cluster_config,
            gates_config=request.gates_config,
            runtime_mode=request.runtime_mode,
            device=request.device,
            precision_mode=request.precision_mode,
            specter_runtime_backend=request.specter_runtime_backend,
            cluster_backend=request.cluster_backend,
            force=request.force,
            progress=request.progress,
            progress_handler=request.progress_handler,
        )
    return request


def run_infer_sources(request: InferSourcesRequest | None = None, **kwargs: Any) -> InferSourcesResult:
    return run_source_inference(_build_infer_request(request, **kwargs))
