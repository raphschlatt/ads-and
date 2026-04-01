from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from author_name_disambiguation.common.pipeline_reports import default_run_id, write_json
from author_name_disambiguation.embedding_contract import DEFAULT_TEXT_MODEL_NAME, build_bundle_embedding_contract
from author_name_disambiguation.hf_transport import embed_texts_via_hf_httpx, normalize_model_name, normalize_provider
from author_name_disambiguation.source_inference import _resolve_model_bundle
from author_name_disambiguation.specter_benchmark import (
    _bucket_counts,
    _build_combined_sources,
    _build_tokenizer,
    _cleanup_output_root_if_empty,
    _collect_hardware_metadata,
    _compute_raw_token_counts_for_frame,
    _cosine_summary,
    _load_normalized_source,
    _make_sample,
    _quantile_summary,
    _resolved_path,
    _top_counts,
    _truncate_text_to_cap,
)

_DEFAULT_REALISTIC_SAMPLE_SIZE = 128
_DEFAULT_MICRO_REPEAT_COUNT = 1000
_DEFAULT_CONCURRENCY_VALUES = (4, 16, 64)
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_BACKOFF_SECONDS = 0.5
_DEFAULT_MAX_BACKOFF_SECONDS = 8.0


@dataclass(slots=True)
class SpecterHFLabBenchmarkRequest:
    publications_path: str | Path
    output_root: str | Path
    dataset_id: str
    model_bundle: str | Path
    references_path: str | Path | None = None
    provider: str = "hf-inference"
    model_name: str = DEFAULT_TEXT_MODEL_NAME
    hf_token_env_var: str = "HF_TOKEN"
    profiles: tuple[str, ...] = ("all",)
    concurrency_values: tuple[int, ...] = _DEFAULT_CONCURRENCY_VALUES
    realistic_sample_size: int = _DEFAULT_REALISTIC_SAMPLE_SIZE
    micro_repeat_count: int = _DEFAULT_MICRO_REPEAT_COUNT
    force: bool = False
    progress: bool = True


@dataclass(slots=True)
class SpecterHFLabBenchmarkResult:
    run_id: str
    output_root: Path
    report_json_path: Path
    report_markdown_path: Path
    summary: str


@dataclass(slots=True)
class _LabDataset:
    name: str
    texts: list[str]
    manifest: list[dict[str, Any]]


@dataclass(slots=True)
class _LabModeSpec:
    name: str
    lab_only: bool
    non_production: bool
    use_http2: bool = False


@dataclass(slots=True)
class _LabRun:
    available: bool
    vectors: np.ndarray
    success_mask: np.ndarray
    errors: list[str | None]
    raw_shapes: list[str | None]
    warmup_seconds: float
    processing_wall_seconds: float
    meta: dict[str, Any]


def _normalize_profiles(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip().lower() for value in values if str(value).strip())
    if not normalized:
        return ("all",)
    if "all" in normalized:
        return ("all",)
    valid = {"prod-safe", "turbo"}
    invalid = sorted(set(normalized) - valid)
    if invalid:
        raise ValueError(f"Unsupported lab benchmark profiles: {invalid!r}")
    return normalized


def _normalize_concurrency_values(values: tuple[int, ...]) -> tuple[int, ...]:
    normalized = sorted({max(1, int(value)) for value in values})
    if not normalized:
        return _DEFAULT_CONCURRENCY_VALUES
    return tuple(normalized)


def _select_mode_specs(profiles: tuple[str, ...]) -> list[_LabModeSpec]:
    if profiles == ("all",):
        return [
            _LabModeSpec("hf_httpx_async_pool_prod_safe", lab_only=True, non_production=False),
            _LabModeSpec(
                "hf_httpx_async_pool_turbo_http2",
                lab_only=True,
                non_production=True,
                use_http2=True,
            ),
        ]
    specs: list[_LabModeSpec] = []
    if "prod-safe" in profiles:
        specs.append(_LabModeSpec("hf_httpx_async_pool_prod_safe", lab_only=True, non_production=False))
    if "turbo" in profiles:
        specs.append(
            _LabModeSpec(
                "hf_httpx_async_pool_turbo_http2",
                lab_only=True,
                non_production=True,
                use_http2=True,
            )
        )
    return specs


def _build_micro_short_repeat_dataset(repeat_count: int) -> _LabDataset:
    text = "Today is a sunny day and I will get some ice cream."
    count = max(1, int(repeat_count))
    return _LabDataset(
        name="micro_short_repeat",
        texts=[text for _ in range(count)],
        manifest=[{"sample_index": idx, "text_kind": "micro_short_repeat"} for idx in range(count)],
    )


def _build_ads_realistic_truncated_dataset(
    *,
    publications_path: str | Path,
    references_path: str | Path | None,
    model_name: str,
    cap: int,
    target_size: int,
) -> _LabDataset:
    tokenizer = _build_tokenizer(model_name)
    publications = _load_normalized_source(publications_path, source_type="publication")
    references = None if references_path is None else _load_normalized_source(references_path, source_type="reference")
    combined = _build_combined_sources(publications, references)
    raw_token_counts = _compute_raw_token_counts_for_frame(combined, tokenizer=tokenizer)
    sample = _make_sample(
        combined,
        name="ads_realistic_truncated",
        target_size=max(1, int(target_size)),
        raw_token_counts_full=raw_token_counts,
    )
    texts: list[str] = []
    manifest: list[dict[str, Any]] = []
    for idx, text in enumerate(sample.texts):
        truncated_text, truncated_tokens, retokenized_tokens = _truncate_text_to_cap(text, tokenizer=tokenizer, cap=cap)
        texts.append(truncated_text)
        row = dict(sample.manifest[idx])
        row["truncated_token_count"] = int(truncated_tokens)
        row["retokenized_token_count"] = int(retokenized_tokens)
        manifest.append(row)
    return _LabDataset(name="ads_realistic_truncated", texts=texts, manifest=manifest)


def _summarize_lab_run(run: _LabRun, *, dataset: _LabDataset) -> dict[str, Any]:
    successful = int(run.success_mask.sum())
    total = len(dataset.texts)
    failed = int(total - successful)
    texts_per_second = None
    if run.processing_wall_seconds > 0 and successful > 0:
        texts_per_second = float(successful / run.processing_wall_seconds)
    return {
        "available": bool(run.available),
        "texts_total": int(total),
        "texts_successful": int(successful),
        "texts_failed": int(failed),
        "success_rate": None if total == 0 else float(successful / max(1, total)),
        "warmup_seconds": float(run.warmup_seconds),
        "processing_wall_seconds": float(run.processing_wall_seconds),
        "texts_per_second": texts_per_second,
        "raw_shape_top_counts": _top_counts(run.raw_shapes),
        "per_text_wall_seconds": _quantile_summary(
            np.full((successful,), float(run.processing_wall_seconds / max(1, successful)), dtype=np.float64)
            if successful > 0
            else np.zeros((0,), dtype=np.float64)
        ),
        "failure_examples": [
            {"sample_index": int(idx), "error": str(run.errors[idx])}
            for idx in range(len(run.errors))
            if run.errors[idx]
        ][:3],
        "meta": dict(run.meta or {}),
    }


def _build_variant_key(mode_name: str, concurrency: int | None) -> str:
    if concurrency is None:
        return mode_name
    return f"{mode_name}__c{int(concurrency)}"


def _run_httpx_mode(
    *,
    dataset: _LabDataset,
    provider: str,
    model_name: str,
    hf_token_env_var: str,
    concurrency: int,
    mode_spec: _LabModeSpec,
) -> _LabRun:
    provider = normalize_provider(provider)
    model_name = normalize_model_name(model_name)
    warmup_seconds = 0.0
    if dataset.texts:
        warmup_started = perf_counter()
        embed_texts_via_hf_httpx(
            texts=[dataset.texts[0]],
            provider=provider,
            model_name=model_name,
            hf_token_env_var=hf_token_env_var,
            max_length=256,
            concurrency=max(1, int(concurrency)),
            use_http2=bool(mode_spec.use_http2),
            chunk_size=1,
            progress=False,
            progress_label=f"{mode_spec.name} warmup",
        )
        warmup_seconds = float(perf_counter() - warmup_started)
    started = perf_counter()
    vectors, meta = embed_texts_via_hf_httpx(
        texts=dataset.texts,
        provider=provider,
        model_name=model_name,
        hf_token_env_var=hf_token_env_var,
        max_length=256,
        concurrency=max(1, int(concurrency)),
        use_http2=bool(mode_spec.use_http2),
        chunk_size=max(64, int(concurrency)),
        progress=False,
        progress_label=mode_spec.name,
    )
    processing_wall_seconds = float(perf_counter() - started)
    return _LabRun(
        available=True,
        vectors=vectors.astype(np.float32, copy=False),
        success_mask=np.ones((len(dataset.texts),), dtype=bool),
        errors=[None] * len(dataset.texts),
        raw_shapes=[None] * len(dataset.texts),
        warmup_seconds=float(warmup_seconds),
        processing_wall_seconds=float(processing_wall_seconds),
        meta={
            "provider": provider,
            "model_name": model_name,
            "transport": str(meta.get("transport") or "httpx_async_pool"),
            "concurrency": int(meta.get("api_concurrency") or concurrency),
            "http2_enabled": meta.get("http2_enabled"),
            "verify": True,
            "lab_only": bool(mode_spec.lab_only),
            "non_production": bool(mode_spec.non_production),
        },
    )


def _run_mode_variant(
    *,
    dataset: _LabDataset,
    mode_spec: _LabModeSpec,
    provider: str,
    model_name: str,
    hf_token_env_var: str,
    concurrency: int | None,
) -> _LabRun:
    if concurrency is None:
        raise ValueError(f"Concurrency is required for HF lab mode {mode_spec.name!r}.")
    return _run_httpx_mode(
        dataset=dataset,
        provider=provider,
        model_name=model_name,
        hf_token_env_var=hf_token_env_var,
        concurrency=int(concurrency),
        mode_spec=mode_spec,
    )


def _build_lab_markdown_report(payload: dict[str, Any], path: Path) -> Path:
    lines = [
        "# SPECTER HF Lab Benchmark Report",
        "",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Bundle: `{payload['model_bundle']}`",
        f"- Summary: `{payload['decision']['summary']}`",
    ]
    for dataset_name, dataset_payload in payload["datasets"].items():
        lines.extend(
            [
                "",
                f"## {dataset_name}",
                "",
                f"- Text count: `{dataset_payload['text_count']}`",
                f"- Best speedup vs prod-safe: `{dataset_payload['best_speedup_vs_prod_safe']}`",
            ]
        )
        for mode_name, mode_payload in dataset_payload["modes"].items():
            best_variant = None
            for variant_name, variant_payload in mode_payload["variants"].items():
                if not variant_payload.get("available"):
                    continue
                if best_variant is None or (variant_payload.get("texts_per_second") or 0.0) > (
                    best_variant.get("texts_per_second") or 0.0
                ):
                    best_variant = variant_payload
            lines.append(
                f"- {mode_name}: "
                f"`available={mode_payload['available']}` "
                f"`lab_only={mode_payload['lab_only']}` "
                f"`non_production={mode_payload['non_production']}` "
                f"`best_tps={None if best_variant is None else best_variant.get('texts_per_second')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_specter_hf_lab_benchmark(request: SpecterHFLabBenchmarkRequest) -> SpecterHFLabBenchmarkResult:
    run_id = default_run_id("specter_hf_lab_benchmark", tag="cli")
    output_root = _resolved_path(request.output_root)
    report_json_path = output_root / "specter_hf_lab_report.json"
    report_markdown_path = output_root / "specter_hf_lab_report.md"
    if output_root.exists() and not request.force and (report_json_path.exists() or report_markdown_path.exists()):
        raise FileExistsError(
            "Refusing to overwrite existing HF lab benchmark artifacts without force=True: "
            f"{report_json_path}, {report_markdown_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        profiles = _normalize_profiles(request.profiles)
        concurrency_values = _normalize_concurrency_values(request.concurrency_values)
        mode_specs = _select_mode_specs(profiles)

        model_info = _resolve_model_bundle(request.model_bundle)
        bundle_contract = dict(
            model_info["manifest"].get("embedding_contract") or build_bundle_embedding_contract(model_info["model_cfg"])
        )
        bundle_text_contract = dict(bundle_contract.get("text", {}) or {})
        model_name = normalize_model_name(request.model_name)
        if bundle_text_contract.get("model_name") and str(bundle_text_contract["model_name"]) != model_name:
            raise ValueError(
                "SPECTER HF lab benchmark must use the bundle text embedding contract. "
                f"bundle={bundle_text_contract.get('model_name')!r} request={model_name!r}"
            )
        token_cap = int(bundle_text_contract.get("tokenization", {}).get("max_length", 256))
        provider = normalize_provider(request.provider)

        datasets = {
            "micro_short_repeat": _build_micro_short_repeat_dataset(int(request.micro_repeat_count)),
            "ads_realistic_truncated": _build_ads_realistic_truncated_dataset(
                publications_path=request.publications_path,
                references_path=request.references_path,
                model_name=model_name,
                cap=int(token_cap),
                target_size=int(request.realistic_sample_size),
            ),
        }

        dataset_payloads: dict[str, Any] = {}
        for dataset_name, dataset in datasets.items():
            mode_payloads: dict[str, Any] = {}
            reference_mode_name = next(
                (mode_spec.name for mode_spec in mode_specs if not bool(mode_spec.non_production)),
                None if not mode_specs else mode_specs[0].name,
            )
            for mode_spec in mode_specs:
                variants: dict[str, Any] = {}
                variant_available = False
                error_message = None
                variant_concurrency_values = concurrency_values
                for concurrency in variant_concurrency_values:
                    variant_key = _build_variant_key(mode_spec.name, concurrency)
                    try:
                        run = _run_mode_variant(
                            dataset=dataset,
                            mode_spec=mode_spec,
                            provider=provider,
                            model_name=model_name,
                            hf_token_env_var=request.hf_token_env_var,
                            concurrency=concurrency,
                        )
                        summary = _summarize_lab_run(run, dataset=dataset)
                        summary["cosine_vs_prod_safe_baseline"] = None
                        variants[variant_key] = summary
                        variant_available = variant_available or bool(run.available)
                    except Exception as exc:
                        error_message = str(exc)
                        variants[variant_key] = {
                            "available": False,
                            "error": str(exc),
                            "texts_total": int(len(dataset.texts)),
                            "texts_successful": 0,
                            "texts_failed": int(len(dataset.texts)),
                            "success_rate": 0.0 if dataset.texts else None,
                            "warmup_seconds": None,
                            "processing_wall_seconds": None,
                            "texts_per_second": None,
                            "raw_shape_top_counts": [],
                            "per_text_wall_seconds": _quantile_summary(np.zeros((0,), dtype=np.float64)),
                            "failure_examples": [],
                            "meta": {
                                "lab_only": bool(mode_spec.lab_only),
                                "non_production": bool(mode_spec.non_production),
                            },
                            "cosine_vs_prod_safe_baseline": None,
                        }
                mode_payloads[mode_spec.name] = {
                    "available": bool(variant_available),
                    "lab_only": bool(mode_spec.lab_only),
                    "non_production": bool(mode_spec.non_production),
                    "error": error_message,
                    "variants": variants,
                }

            reference_tps = None
            if reference_mode_name and reference_mode_name in mode_payloads:
                for variant_payload in mode_payloads[reference_mode_name]["variants"].values():
                    tps = variant_payload.get("texts_per_second")
                    if isinstance(tps, (int, float)):
                        reference_tps = float(tps) if reference_tps is None else max(reference_tps, float(tps))

            best_speedup = None
            for mode_payload in mode_payloads.values():
                for variant_payload in mode_payload["variants"].values():
                    tps = variant_payload.get("texts_per_second")
                    speedup = (
                        None
                        if reference_tps is None
                        or not isinstance(tps, (int, float))
                        or float(tps) <= 0
                        else float(float(tps) / float(reference_tps))
                    )
                    variant_payload["speedup_vs_prod_safe"] = speedup
                    if isinstance(speedup, (int, float)):
                        best_speedup = float(speedup) if best_speedup is None else max(best_speedup, float(speedup))
            dataset_payloads[dataset_name] = {
                "text_count": int(len(dataset.texts)),
                "sample_manifest": list(dataset.manifest[: min(16, len(dataset.manifest))]),
                "modes": mode_payloads,
                "reference_mode_name": reference_mode_name,
                "best_speedup_vs_prod_safe": best_speedup,
            }

        micro_speedup = dataset_payloads["micro_short_repeat"].get("best_speedup_vs_prod_safe")
        realistic_speedup = dataset_payloads["ads_realistic_truncated"].get("best_speedup_vs_prod_safe")
        decision_summary = (
            "HF lab benchmark completed. "
            f"best_micro_speedup_vs_prod_safe={micro_speedup}; "
            f"best_ads_speedup_vs_prod_safe={realistic_speedup}"
        )
        payload = {
            "run_id": run_id,
            "generated_utc": pd.Timestamp.utcnow().isoformat(),
            "dataset_id": str(request.dataset_id),
            "model_bundle": str(_resolved_path(request.model_bundle)),
            "provider": provider,
            "model_name": model_name,
            "token_cap": int(token_cap),
            "profiles": list(profiles),
            "concurrency_values": [int(value) for value in concurrency_values],
            "hardware": _collect_hardware_metadata(),
            "datasets": dataset_payloads,
            "decision": {
                "summary": decision_summary,
                "best_micro_short_repeat_speedup_vs_prod_safe": micro_speedup,
                "best_ads_realistic_truncated_speedup_vs_prod_safe": realistic_speedup,
                "ten_x_or_more_observed": bool(
                    (isinstance(micro_speedup, (int, float)) and float(micro_speedup) >= 10.0)
                    or (isinstance(realistic_speedup, (int, float)) and float(realistic_speedup) >= 10.0)
                ),
            },
            "artifacts": {
                "output_root": str(output_root),
                "report_json_path": str(report_json_path),
                "report_markdown_path": str(report_markdown_path),
            },
        }
        write_json(payload, report_json_path)
        _build_lab_markdown_report(payload, report_markdown_path)
        return SpecterHFLabBenchmarkResult(
            run_id=run_id,
            output_root=output_root,
            report_json_path=report_json_path,
            report_markdown_path=report_markdown_path,
            summary=decision_summary,
        )
    except Exception:
        if report_json_path.exists():
            report_json_path.unlink()
        if report_markdown_path.exists():
            report_markdown_path.unlink()
        raise
    finally:
        _cleanup_output_root_if_empty(output_root)
