from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from author_name_disambiguation.source_inference import run_source_inference


UidScope = Literal["dataset", "local", "registry"]
InferStage = Literal["smoke", "mini", "mid", "full"]
PrecisionMode = Literal["fp32", "amp_bf16"]
ClusterBackend = Literal["auto", "sklearn_cpu", "cuml_gpu"]
SpecterRuntimeBackend = Literal["transformers", "onnx_fp32"]
RuntimeMode = Literal["gpu", "cpu", "hf"]


@dataclass(slots=True)
class InferSourcesRequest:
    publications_path: str | Path
    output_root: str | Path
    dataset_id: str
    model_bundle: str | Path
    references_path: str | Path | None = None
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


def run_infer_sources(request: InferSourcesRequest) -> InferSourcesResult:
    return run_source_inference(request)
