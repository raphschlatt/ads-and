from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


UidScope = Literal["dataset", "local", "registry"]


@dataclass
class InferAdsRequest:
    dataset_id: str
    model_run_id: str | None = None
    model_bundle: str | Path | None = None
    infer_stage: Literal["smoke", "mini", "mid", "full"] = "full"
    infer_run_config: str | None = None
    paths_config: str = "configs/paths.local.yaml"
    cluster_config: str = "configs/clustering/dbscan_paper.yaml"
    gates_config: str | None = "configs/gates.yaml"
    run_id: str | None = None
    device: str = "auto"
    precision_mode: Literal["fp32", "amp_bf16"] = "fp32"
    score_batch_size: int | None = None
    memory_policy: Literal["fail", "warn"] = "fail"
    max_ram_fraction: float = 0.80
    cpu_sharding: Literal["auto", "on", "off"] | None = None
    cpu_workers: str | int | None = None
    cpu_min_pairs_per_worker: int | None = None
    cpu_target_ram_fraction: float | None = None
    cluster_backend: Literal["auto", "sklearn_cpu", "cuml_gpu"] | None = None
    uid_scope: UidScope = "dataset"
    uid_namespace: str | None = None
    baseline_run_id: str | None = None
    force: bool = False
    progress: bool = True
    quiet_libs: bool = True


@dataclass
class InferAdsResult:
    run_id: str
    go: bool | None
    metrics_dir: Path
    clusters_path: Path
    publication_authors_path: Path
    publications_disambiguated_path: Path
    references_disambiguated_path: Path | None
    stage_metrics_path: Path
    go_no_go_path: Path


def _validate_model_source(request: InferAdsRequest) -> None:
    has_run_id = request.model_run_id is not None
    has_bundle = request.model_bundle is not None
    if has_run_id == has_bundle:
        raise ValueError("Provide exactly one of model_run_id or model_bundle.")


def run_infer_ads(request: InferAdsRequest) -> InferAdsResult:
    _validate_model_source(request)

    from src import cli

    args = Namespace(
        dataset_id=str(request.dataset_id),
        model_run_id=None if request.model_run_id is None else str(request.model_run_id),
        model_bundle=None if request.model_bundle is None else str(request.model_bundle),
        infer_stage=str(request.infer_stage),
        infer_run_config=request.infer_run_config,
        paths_config=str(request.paths_config),
        cluster_config=str(request.cluster_config),
        gates_config=request.gates_config,
        run_id=request.run_id,
        device=str(request.device),
        precision_mode=str(request.precision_mode),
        score_batch_size=request.score_batch_size,
        memory_policy=str(request.memory_policy),
        max_ram_fraction=float(request.max_ram_fraction),
        cpu_sharding=request.cpu_sharding,
        cpu_workers=request.cpu_workers,
        cpu_min_pairs_per_worker=request.cpu_min_pairs_per_worker,
        cpu_target_ram_fraction=request.cpu_target_ram_fraction,
        cluster_backend=request.cluster_backend,
        uid_scope=str(request.uid_scope),
        uid_namespace=request.uid_namespace,
        baseline_run_id=request.baseline_run_id,
        force=bool(request.force),
        progress=bool(request.progress),
        quiet_libs=bool(request.quiet_libs),
    )

    payload = cli._run_infer_ads_impl(args)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected infer payload; expected dict.")

    return InferAdsResult(
        run_id=str(payload["run_id"]),
        go=payload.get("go"),
        metrics_dir=Path(str(payload["metrics_dir"])),
        clusters_path=Path(str(payload["clusters_path"])),
        publication_authors_path=Path(str(payload["publication_authors_path"])),
        publications_disambiguated_path=Path(str(payload["publications_disambiguated_path"])),
        references_disambiguated_path=(
            None if payload.get("references_disambiguated_path") is None else Path(str(payload["references_disambiguated_path"]))
        ),
        stage_metrics_path=Path(str(payload["stage_metrics_path"])),
        go_no_go_path=Path(str(payload["go_no_go_path"])),
    )
