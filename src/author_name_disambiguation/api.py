from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from author_name_disambiguation.common.cli_ui import CliUI, get_active_ui
from author_name_disambiguation.defaults import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_RAW_LSPO_PARQUET,
    FIXED_MODEL_BASELINE_RUN_ID,
    resolve_fixed_model_bundle_path,
)
from author_name_disambiguation.infer_sources import InferSourcesResult, run_infer_sources
from author_name_disambiguation.workflow_helpers import (
    default_train_run_id,
    resolve_report_paths,
    sanitize_report_tag,
)


UserRuntime = Literal["auto", "gpu", "cpu", "hf"]
UserInferStage = Literal["full", "incremental", "smoke", "mini", "mid"]
TrainStage = Literal["smoke", "mini", "mid", "full"]
ProgressStyle = Literal["compact", "verbose"]


@dataclass(slots=True)
class LspoQualityResult:
    model_run_id: str
    metrics_dir: Path
    report_json_path: Path
    summary_csv_path: Path
    per_seed_csv_path: Path
    report_markdown_path: Path


@dataclass(slots=True)
class LspoTrainingResult:
    run_id: str
    metrics_dir: Path
    train_manifest_path: Path
    stage_metrics_path: Path
    go_no_go_path: Path
    cluster_config_used_path: Path


def _load_cli_module():
    from author_name_disambiguation import cli

    return cli


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
    if normalized not in {"gpu", "cpu", "hf"}:
        raise ValueError(f"Unsupported runtime={runtime!r}. Expected one of ['auto', 'cpu', 'gpu', 'hf'].")
    return normalized


def _derive_model_run_id(*, model_run_id: str | None, model_bundle: str | Path | None) -> str:
    if model_run_id is not None and str(model_run_id).strip():
        return str(model_run_id).strip()
    if model_bundle is None:
        return FIXED_MODEL_BASELINE_RUN_ID
    bundle_path = Path(model_bundle).expanduser().resolve()
    if bundle_path.name == "bundle_v1":
        parent_name = bundle_path.parent.name.strip()
        if parent_name:
            return parent_name
    raise ValueError(
        "model_run_id is required unless model_bundle points to artifacts/models/<run_id>/bundle_v1 "
        "or the packaged Fixed Model Baseline is used."
    )


def disambiguate_sources(
    publications_path: str | Path,
    *,
    references_path: str | Path | None = None,
    output_dir: str | Path,
    runtime: UserRuntime = "auto",
    dataset_id: str | None = None,
    force: bool = False,
    model_bundle: str | Path | None = None,
    infer_stage: UserInferStage = "full",
    progress: bool = True,
    progress_style: ProgressStyle = "compact",
) -> InferSourcesResult:
    resolved_runtime = _resolve_user_runtime(runtime)
    resolved_dataset_id = _derive_dataset_id(
        publications_path=publications_path,
        output_dir=output_dir,
        dataset_id=dataset_id,
    )
    resolved_bundle = resolve_fixed_model_bundle_path() if model_bundle is None else Path(model_bundle).expanduser()
    created_ui = None
    if get_active_ui() is None:
        created_ui = CliUI(total_steps=8, progress=bool(progress), progress_style=str(progress_style))
    try:
        return run_infer_sources(
            publications_path=publications_path,
            references_path=references_path,
            output_root=output_dir,
            dataset_id=resolved_dataset_id,
            model_bundle=resolved_bundle,
            infer_stage=infer_stage,
            runtime_mode=resolved_runtime,
            force=bool(force),
            progress=bool(progress),
        )
    finally:
        if created_ui is not None:
            created_ui.close()


def evaluate_lspo_quality(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    raw_lspo_parquet: str | Path = DEFAULT_RAW_LSPO_PARQUET,
    raw_lspo_h5: str | Path | None = None,
    model_run_id: str | None = None,
    model_bundle: str | Path | None = None,
    report_tag: str | None = None,
    allow_legacy_lspo_compat: bool = True,
    device: str = "auto",
    precision_mode: str = "fp32",
    score_batch_size: int = 8192,
    force: bool = False,
    progress: bool = True,
    progress_style: ProgressStyle = "compact",
    quiet_libs: bool = True,
) -> LspoQualityResult:
    resolved_model_run_id = _derive_model_run_id(model_run_id=model_run_id, model_bundle=model_bundle)
    args = SimpleNamespace(
        model_run_id=resolved_model_run_id,
        data_root=str(Path(data_root).expanduser()),
        artifacts_root=str(Path(artifacts_root).expanduser()),
        raw_lspo_parquet=str(Path(raw_lspo_parquet).expanduser()),
        raw_lspo_h5=None if raw_lspo_h5 is None else str(Path(raw_lspo_h5).expanduser()),
        device=str(device),
        precision_mode=str(precision_mode),
        score_batch_size=int(score_batch_size),
        cluster_config_override=None,
        report_tag=report_tag,
        allow_legacy_lspo_compat=bool(allow_legacy_lspo_compat),
        force=bool(force),
        progress=bool(progress),
        progress_style=str(progress_style),
        quiet_libs=bool(quiet_libs),
    )
    _load_cli_module().cmd_run_cluster_test_report(args)
    metrics_dir = Path(args.artifacts_root).expanduser().resolve() / "metrics" / resolved_model_run_id
    report_paths = resolve_report_paths(metrics_dir, report_tag=sanitize_report_tag(report_tag))
    return LspoQualityResult(
        model_run_id=resolved_model_run_id,
        metrics_dir=metrics_dir,
        report_json_path=report_paths["json"],
        summary_csv_path=report_paths["summary_csv"],
        per_seed_csv_path=report_paths["per_seed_csv"],
        report_markdown_path=report_paths["markdown"],
    )


def train_lspo_model(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    raw_lspo_parquet: str | Path = DEFAULT_RAW_LSPO_PARQUET,
    raw_lspo_h5: str | Path | None = None,
    run_stage: TrainStage = "full",
    run_id: str | None = None,
    run_config: str | Path | None = None,
    model_config: str | Path | None = None,
    cluster_config: str | Path | None = None,
    gates_config: str | Path | None = None,
    device: str = "auto",
    precision_mode: str | None = None,
    seeds: list[int] | None = None,
    baseline_run_id: str | None = None,
    score_batch_size: int = 8192,
    force: bool = False,
    progress: bool = True,
    progress_style: ProgressStyle = "compact",
    quiet_libs: bool = True,
) -> LspoTrainingResult:
    resolved_run_id = run_id or default_train_run_id(run_stage)
    args = SimpleNamespace(
        run_stage=str(run_stage),
        data_root=str(Path(data_root).expanduser()),
        artifacts_root=str(Path(artifacts_root).expanduser()),
        raw_lspo_parquet=str(Path(raw_lspo_parquet).expanduser()),
        raw_lspo_h5=None if raw_lspo_h5 is None else str(Path(raw_lspo_h5).expanduser()),
        run_config=None if run_config is None else str(Path(run_config).expanduser()),
        model_config=None if model_config is None else str(Path(model_config).expanduser()),
        cluster_config=None if cluster_config is None else str(Path(cluster_config).expanduser()),
        gates_config=None if gates_config is None else str(Path(gates_config).expanduser()),
        run_id=resolved_run_id,
        device=str(device),
        precision_mode=precision_mode,
        seeds=seeds,
        use_stub_embeddings=False,
        force=bool(force),
        baseline_run_id=baseline_run_id,
        score_batch_size=int(score_batch_size),
        progress=bool(progress),
        progress_style=str(progress_style),
        quiet_libs=bool(quiet_libs),
    )
    _load_cli_module().cmd_run_train_stage(args)
    metrics_dir = Path(args.artifacts_root).expanduser().resolve() / "metrics" / resolved_run_id
    stage = str(run_stage).strip().lower()
    return LspoTrainingResult(
        run_id=resolved_run_id,
        metrics_dir=metrics_dir,
        train_manifest_path=metrics_dir / "03_train_manifest.json",
        stage_metrics_path=metrics_dir / f"05_stage_metrics_{stage}.json",
        go_no_go_path=metrics_dir / f"05_go_no_go_{stage}.json",
        cluster_config_used_path=metrics_dir / "04_clustering_config_used.json",
    )
