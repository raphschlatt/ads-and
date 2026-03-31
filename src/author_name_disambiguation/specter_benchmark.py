from __future__ import annotations

import asyncio
from dataclasses import dataclass
import ast
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import numpy as np
import pandas as pd

from author_name_disambiguation.common.cli_ui import loop_progress
from author_name_disambiguation.common.pipeline_reports import default_run_id, write_json
from author_name_disambiguation.data.prepare_ads import load_ads_records, normalize_ads_mentions
from author_name_disambiguation.embedding_contract import (
    DEFAULT_TEXT_MODEL_NAME,
    build_bundle_embedding_contract,
    build_source_text,
)
from author_name_disambiguation.hf_compatibility_report import _compare_infer_outputs, _standardize_sample_frame
from author_name_disambiguation.infer_sources import InferSourcesRequest, run_infer_sources
from author_name_disambiguation.precompute_source_embeddings import (
    _is_retryable_hf_error,
    _normalize_hf_batch_response,
    _normalize_model_name,
    _normalize_provider,
    _resolve_hf_token,
)
from author_name_disambiguation.source_inference import _estimate_pair_upper_bound, _resolve_infer_run_cfg, _resolve_model_bundle

_TRACK_A_CAP = 512
_DEFAULT_TRACK_B_CAP = 256
_DEFAULT_PARITY_SAMPLE_SIZE = 128
_DEFAULT_THROUGHPUT_SAMPLE_SIZE = 2048
_DEFAULT_API_CONCURRENCY = 4
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BASE_BACKOFF_SECONDS = 1.0
_DEFAULT_MAX_BACKOFF_SECONDS = 30.0


@dataclass(slots=True)
class SpecterBenchmarkRequest:
    publications_path: str | Path
    output_root: str | Path
    dataset_id: str
    model_bundle: str | Path
    references_path: str | Path | None = None
    provider: str = "hf-inference"
    model_name: str = DEFAULT_TEXT_MODEL_NAME
    hf_token_env_var: str = "HF_TOKEN"
    parity_sample_size: int = _DEFAULT_PARITY_SAMPLE_SIZE
    throughput_sample_size: int = _DEFAULT_THROUGHPUT_SAMPLE_SIZE
    local_batch_size: int | None = None
    cpu_device: str = "cpu"
    gpu_device: str = "cuda"
    api_concurrency: int = _DEFAULT_API_CONCURRENCY
    force: bool = False
    progress: bool = True


@dataclass(slots=True)
class SpecterBenchmarkResult:
    run_id: str
    output_root: Path
    report_json_path: Path
    report_markdown_path: Path
    recommendation: str


@dataclass(slots=True)
class _BenchmarkSample:
    name: str
    frame: pd.DataFrame
    texts: list[str]
    raw_token_counts: np.ndarray
    manifest: list[dict[str, Any]]


@dataclass(slots=True)
class _ModeRun:
    mode: str
    available: bool
    vectors: np.ndarray
    attempted_mask: np.ndarray
    success_mask: np.ndarray
    per_item_wall_seconds: np.ndarray
    raw_shapes: list[str | None]
    sent_token_counts: np.ndarray
    errors: list[str | None]
    load_seconds: float
    processing_wall_seconds: float
    total_wall_seconds: float
    meta: dict[str, Any]


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_normalized_source(path: str | Path, *, source_type: str) -> pd.DataFrame:
    return load_ads_records(path, source_type=source_type).reset_index(drop=True)


def _sample_indices(total: int, target: int) -> list[int]:
    if total <= 0:
        return []
    sample_n = min(int(max(1, target)), total)
    if sample_n >= total:
        return list(range(total))
    raw = np.linspace(0, total - 1, num=sample_n)
    seen: set[int] = set()
    out: list[int] = []
    for value in raw:
        idx = min(total - 1, max(0, int(round(float(value)))))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if len(out) < sample_n:
        for idx in range(total):
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
            if len(out) >= sample_n:
                break
    return sorted(out)


def _build_combined_sources(publications: pd.DataFrame, references: pd.DataFrame | None) -> pd.DataFrame:
    pub = publications.copy()
    pub["_benchmark_source"] = "publications"
    pub["_benchmark_source_order"] = np.arange(len(pub), dtype=np.int64)
    frames = [pub]
    if references is not None:
        ref = references.copy()
        ref["_benchmark_source"] = "references"
        ref["_benchmark_source_order"] = np.arange(len(ref), dtype=np.int64)
        frames.append(ref)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined.reset_index(drop=True)


def _extract_constant_assignments(code: str) -> dict[str, str]:
    out: dict[str, str] = {}
    tree = ast.parse(code)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"title", "abstract"}:
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str):
            out[target.id] = value
    return out


def _load_notebook_mwe_text(notebook_path: str | Path | None = None) -> dict[str, Any]:
    path = _resolved_path(notebook_path or (_repo_root() / "Test.ipynb"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    title = ""
    abstract = ""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or [])
        if "title =" not in source or "abstract =" not in source:
            continue
        values = _extract_constant_assignments(source)
        title = values.get("title", title)
        abstract = values.get("abstract", abstract)
        if title and abstract:
            break
    if not title and not abstract:
        raise RuntimeError(f"Could not extract title/abstract MWE from {path}")
    text = build_source_text(title, abstract)
    return {
        "notebook_path": str(path),
        "title": title,
        "abstract": abstract,
        "text": text,
    }


def _build_texts_from_frame(frame: pd.DataFrame) -> list[str]:
    if len(frame) == 0:
        return []
    titles = frame["title"].fillna("").astype(str).tolist()
    abstracts = frame["abstract"].fillna("").astype(str).tolist()
    return [build_source_text(title, abstract) for title, abstract in zip(titles, abstracts, strict=True)]


def _build_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _compute_raw_token_counts_for_texts(
    texts: list[str],
    *,
    tokenizer: Any,
    chunk_size: int = 1024,
) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.int32)
    lengths: list[int] = []
    for start in range(0, len(texts), max(1, int(chunk_size))):
        batch = texts[start : start + max(1, int(chunk_size))]
        enc = tokenizer(batch, padding=False, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(lengths, dtype=np.int32)


def _compute_raw_token_counts_for_frame(
    frame: pd.DataFrame,
    *,
    tokenizer: Any,
    chunk_size: int = 1024,
) -> np.ndarray:
    if len(frame) == 0:
        return np.zeros((0,), dtype=np.int32)
    lengths: list[int] = []
    titles = frame["title"].fillna("").astype(str).tolist()
    abstracts = frame["abstract"].fillna("").astype(str).tolist()
    size = max(1, int(chunk_size))
    for start in range(0, len(frame), size):
        batch_titles = titles[start : start + size]
        batch_abstracts = abstracts[start : start + size]
        texts = [build_source_text(title, abstract) for title, abstract in zip(batch_titles, batch_abstracts, strict=True)]
        enc = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(lengths, dtype=np.int32)


def _summarize_token_lengths(lengths: np.ndarray) -> dict[str, Any]:
    if lengths.size == 0:
        return {
            "count": 0,
            "max": 0,
            "quantiles": {},
        }
    quantiles = {
        "0.5": float(np.quantile(lengths, 0.5)),
        "0.9": float(np.quantile(lengths, 0.9)),
        "0.95": float(np.quantile(lengths, 0.95)),
        "0.99": float(np.quantile(lengths, 0.99)),
        "0.999": float(np.quantile(lengths, 0.999)),
    }
    return {
        "count": int(lengths.size),
        "max": int(lengths.max()),
        "quantiles": quantiles,
    }


def _bucket_spec(cap: int) -> list[tuple[str, Any]]:
    return [
        (f"<= {cap}", lambda n, bound=cap: int(n) <= int(bound)),
        (f"{cap + 1}-512", lambda n, bound=cap: int(bound) < int(n) <= 512),
        ("> 512", lambda n: int(n) > 512),
    ]


def _bucket_counts(lengths: np.ndarray, cap: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, predicate in _bucket_spec(cap):
        counts[label] = int(sum(1 for value in lengths if predicate(int(value))))
    return counts


def _make_sample(combined: pd.DataFrame, *, name: str, target_size: int, raw_token_counts_full: np.ndarray) -> _BenchmarkSample:
    indices = _sample_indices(len(combined), target_size)
    sampled = combined.iloc[indices].copy().reset_index(drop=True) if indices else combined.iloc[0:0].copy()
    texts = _build_texts_from_frame(sampled)
    raw_counts = raw_token_counts_full[np.asarray(indices, dtype=np.int64)] if indices else np.zeros((0,), dtype=np.int32)
    manifest: list[dict[str, Any]] = []
    for sample_index, (_, row) in enumerate(sampled.iterrows()):
        manifest.append(
            {
                "sample_index": int(sample_index),
                "source": str(row.get("_benchmark_source", "")),
                "source_order": int(row.get("_benchmark_source_order", sample_index)),
                "bibcode": str(row.get("bibcode", "")),
                "year": None if pd.isna(row.get("year")) else int(row.get("year")),
                "raw_token_count": int(raw_counts[sample_index]) if sample_index < len(raw_counts) else 0,
            }
        )
    return _BenchmarkSample(
        name=str(name),
        frame=sampled,
        texts=texts,
        raw_token_counts=raw_counts.astype(np.int32, copy=False),
        manifest=manifest,
    )


def _collect_hardware_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        info["cpu_count_physical"] = None
        info["ram_bytes"] = None

    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["gpu_name"] = None

    try:
        import transformers

        info["transformers_version"] = str(transformers.__version__)
    except Exception:
        info["transformers_version"] = None

    try:
        import huggingface_hub

        info["huggingface_hub_version"] = str(huggingface_hub.__version__)
    except Exception:
        info["huggingface_hub_version"] = None
    return info


def _truncate_text_to_cap(text: str, *, tokenizer: Any, cap: int) -> tuple[str, int, int]:
    truncated = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=int(cap),
        add_special_tokens=True,
    )
    truncated_text = tokenizer.decode(truncated["input_ids"], skip_special_tokens=True)
    retokenized = tokenizer(truncated_text, padding=False, truncation=False, add_special_tokens=True)
    return truncated_text, int(len(truncated["input_ids"])), int(len(retokenized["input_ids"]))


def _quantile_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.quantile(finite, 0.5)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _top_counts(values: list[str | None], *, limit: int = 5) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    rows = [{"value": key, "count": int(count)} for key, count in counts.items()]
    rows.sort(key=lambda row: (-int(row["count"]), str(row["value"])))
    return rows[:limit]


def _cosine_summary(
    vectors: np.ndarray,
    reference_vectors: np.ndarray,
    *,
    mask: np.ndarray,
) -> dict[str, Any]:
    if vectors.shape != reference_vectors.shape:
        raise ValueError(f"Cosine compare shape mismatch: {vectors.shape} vs {reference_vectors.shape}")
    indices = np.where(mask)[0]
    if indices.size == 0:
        return {"compared_count": 0, "mean_cosine_similarity": None, "min_cosine_similarity": None}
    lhs = vectors[indices].astype(np.float32, copy=False)
    rhs = reference_vectors[indices].astype(np.float32, copy=False)
    lhs_norm = np.linalg.norm(lhs, axis=1, keepdims=True).clip(min=1e-8)
    rhs_norm = np.linalg.norm(rhs, axis=1, keepdims=True).clip(min=1e-8)
    cosines = ((lhs / lhs_norm) * (rhs / rhs_norm)).sum(axis=1)
    return {
        "compared_count": int(indices.size),
        "mean_cosine_similarity": float(np.mean(cosines)),
        "min_cosine_similarity": float(np.min(cosines)),
    }


def _failure_examples(
    run: _ModeRun,
    sample: _BenchmarkSample,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(sample.texts)):
        if not run.attempted_mask[idx] or run.success_mask[idx]:
            continue
        meta = dict(sample.manifest[idx])
        meta["error"] = run.errors[idx]
        meta["sent_token_count"] = int(run.sent_token_counts[idx]) if np.isfinite(run.sent_token_counts[idx]) else None
        meta["wall_seconds"] = (
            float(run.per_item_wall_seconds[idx]) if np.isfinite(run.per_item_wall_seconds[idx]) else None
        )
        rows.append(meta)
        if len(rows) >= limit:
            break
    return rows


def _summarize_mode_run(
    run: _ModeRun,
    sample: _BenchmarkSample,
    *,
    cap: int,
    reference_vectors: np.ndarray | None = None,
    reference_success_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    total = len(sample.texts)
    attempted = int(run.attempted_mask.sum())
    successful = int(run.success_mask.sum())
    skipped = int(total - attempted)
    failed = int(attempted - successful)
    throughput = None
    if run.processing_wall_seconds > 0 and successful > 0:
        throughput = float(successful / run.processing_wall_seconds)
    summary = {
        "mode": run.mode,
        "available": bool(run.available),
        "texts_total": int(total),
        "texts_attempted": int(attempted),
        "texts_successful": int(successful),
        "texts_failed": int(failed),
        "texts_skipped": int(skipped),
        "success_rate_attempted": None if attempted == 0 else float(successful / attempted),
        "success_rate_full_sample": None if total == 0 else float(successful / max(1, total)),
        "load_seconds": float(run.load_seconds),
        "processing_wall_seconds": float(run.processing_wall_seconds),
        "total_wall_seconds": float(run.total_wall_seconds),
        "texts_per_second": throughput,
        "per_text_wall_seconds": _quantile_summary(run.per_item_wall_seconds[run.success_mask]),
        "raw_shape_top_counts": _top_counts(run.raw_shapes),
        "failure_examples": _failure_examples(run, sample),
        "meta": dict(run.meta or {}),
        "buckets": {},
        "cosine_vs_local_gpu": None,
    }
    if reference_vectors is not None and reference_success_mask is not None:
        summary["cosine_vs_local_gpu"] = _cosine_summary(
            run.vectors,
            reference_vectors,
            mask=(run.success_mask & reference_success_mask),
        )

    for label, predicate in _bucket_spec(cap):
        bucket_mask = np.asarray([predicate(int(value)) for value in sample.raw_token_counts], dtype=bool)
        bucket_attempted = run.attempted_mask & bucket_mask
        bucket_success = run.success_mask & bucket_mask
        bucket_summary: dict[str, Any] = {
            "texts_total": int(bucket_mask.sum()),
            "texts_attempted": int(bucket_attempted.sum()),
            "texts_successful": int(bucket_success.sum()),
            "texts_skipped": int((bucket_mask & ~run.attempted_mask).sum()),
            "success_rate_full_bucket": None
            if int(bucket_mask.sum()) == 0
            else float(bucket_success.sum() / max(1, int(bucket_mask.sum()))),
            "per_text_wall_seconds": _quantile_summary(run.per_item_wall_seconds[bucket_success]),
        }
        if reference_vectors is not None and reference_success_mask is not None:
            bucket_summary["cosine_vs_local_gpu"] = _cosine_summary(
                run.vectors,
                reference_vectors,
                mask=(bucket_success & reference_success_mask),
            )
        summary["buckets"][label] = bucket_summary
    return summary


def _hf_http2_enabled() -> bool:
    return importlib.util.find_spec("h2") is not None


def _build_hf_feature_extraction_url(*, provider: str, model_name: str) -> str:
    from huggingface_hub import constants

    base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider=provider)
    return f"{base_url}/models/{model_name}/pipeline/feature-extraction"


async def _request_hf_single_async(
    *,
    client: httpx.AsyncClient,
    url: str,
    text: str,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_backoff_seconds: float = _DEFAULT_BASE_BACKOFF_SECONDS,
    max_backoff_seconds: float = _DEFAULT_MAX_BACKOFF_SECONDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    started_at = perf_counter()
    attempts = 0
    backoff_seconds_total = 0.0
    request_payload = {
        "inputs": text,
        "parameters": {
            "truncate": True,
            "truncation_direction": "Right",
        },
    }
    while True:
        attempts += 1
        try:
            response = await client.post(url, json=request_payload)
            response.raise_for_status()
            payload = response.json()
            arr = np.asarray(payload, dtype=np.float32)
            vector = _normalize_hf_batch_response(arr, expected_items=1)[0]
            return vector.astype(np.float32, copy=False), {
                "attempts": int(attempts),
                "backoff_seconds_total": float(backoff_seconds_total),
                "wall_seconds": float(perf_counter() - started_at),
                "raw_shape": [int(v) for v in arr.shape],
            }
        except Exception as exc:
            should_retry = attempts <= int(max_retries) and _is_retryable_hf_error(exc)
            if not should_retry:
                detail = str(exc)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    body = exc.response.text.strip()
                    if body:
                        detail = f"HTTP {exc.response.status_code}: {body[:240]}"
                raise RuntimeError(
                    f"HF feature_extraction failed after {attempts} attempt(s): {type(exc).__name__}: {detail}"
                ) from exc
            delay = min(float(max_backoff_seconds), float(base_backoff_seconds) * (2 ** (attempts - 1)))
            await asyncio.sleep(delay)
            backoff_seconds_total += float(delay)


def _run_hf_mode(
    *,
    sample: _BenchmarkSample,
    mode_name: str,
    model_name: str,
    provider: str,
    hf_token_env_var: str,
    tokenizer: Any,
    cap: int | None = None,
    progress: bool,
    api_concurrency: int = _DEFAULT_API_CONCURRENCY,
) -> _ModeRun:
    texts_total = len(sample.texts)
    vectors = np.full((texts_total, 768), np.nan, dtype=np.float32)
    attempted_mask = np.zeros((texts_total,), dtype=bool)
    success_mask = np.zeros((texts_total,), dtype=bool)
    per_item_wall_seconds = np.full((texts_total,), np.nan, dtype=np.float64)
    sent_token_counts = np.full((texts_total,), np.nan, dtype=np.float64)
    raw_shapes: list[str | None] = [None] * texts_total
    errors: list[str | None] = [None] * texts_total

    provider = _normalize_provider(provider)
    model_name = _normalize_model_name(model_name)
    api_key = _resolve_hf_token(hf_token_env_var)
    effective_concurrency = max(1, int(api_concurrency))
    http2_enabled = _hf_http2_enabled()
    request_url = _build_hf_feature_extraction_url(provider=provider, model_name=model_name)

    def _prepare_text(idx: int) -> tuple[str, int]:
        raw_tokens = int(sample.raw_token_counts[idx])
        if cap is None:
            return sample.texts[idx], raw_tokens
        sent_text, _truncated_input_ids, retokenized_tokens = _truncate_text_to_cap(
            sample.texts[idx],
            tokenizer=tokenizer,
            cap=int(cap),
        )
        return sent_text, int(retokenized_tokens)

    async def _run_all(tracker: Any) -> None:
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
        limits = httpx.Limits(
            max_connections=int(effective_concurrency),
            max_keepalive_connections=int(effective_concurrency),
        )
        headers = {"Authorization": f"Bearer {api_key}"}
        semaphore = asyncio.Semaphore(int(effective_concurrency))

        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=limits,
            http2=bool(http2_enabled),
        ) as client:

            async def _run_one(idx: int) -> None:
                attempted_mask[idx] = True
                sent_text, sent_token_count = _prepare_text(idx)
                try:
                    async with semaphore:
                        vector, meta = await _request_hf_single_async(
                            client=client,
                            url=request_url,
                            text=str(sent_text),
                        )
                    vectors[idx] = vector
                    success_mask[idx] = True
                    per_item_wall_seconds[idx] = float(meta.get("wall_seconds") or 0.0)
                    sent_token_counts[idx] = float(sent_token_count)
                    raw_shapes[idx] = None if meta.get("raw_shape") is None else str(meta["raw_shape"])
                except Exception as exc:
                    errors[idx] = str(exc)
                finally:
                    tracker.update(1)

            await asyncio.gather(*(_run_one(idx) for idx in range(texts_total)))

    started_at = perf_counter()
    with loop_progress(
        total=texts_total,
        label=f"{mode_name} {sample.name}",
        enabled=bool(progress),
        unit="text",
    ) as tracker:
        asyncio.run(_run_all(tracker))

    total_wall_seconds = perf_counter() - started_at
    return _ModeRun(
        mode=str(mode_name),
        available=True,
        vectors=vectors,
        attempted_mask=attempted_mask,
        success_mask=success_mask,
        per_item_wall_seconds=per_item_wall_seconds,
        raw_shapes=raw_shapes,
        sent_token_counts=sent_token_counts,
        errors=errors,
        load_seconds=0.0,
        processing_wall_seconds=float(total_wall_seconds),
        total_wall_seconds=float(total_wall_seconds),
        meta={
            "provider": provider,
            "model_name": model_name,
            "transport": "httpx_async_pool",
            "api_concurrency": int(effective_concurrency),
            "http2_enabled": bool(http2_enabled),
            "client_truncation_cap": None if cap is None else int(cap),
        },
    )


class _LocalSpecterSession:
    def __init__(self, *, model_name: str, device: str):
        started_at = perf_counter()
        from transformers import AutoModel, AutoTokenizer

        self.device = str(device)
        self.model_name = str(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        import torch

        self.torch = torch
        self.available = True
        self.load_seconds = 0.0
        try:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in this environment.")
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.available = False
            raise
        finally:
            self.load_seconds = float(perf_counter() - started_at)

    def run(self, *, sample: _BenchmarkSample, cap: int, batch_size: int | None, progress: bool) -> _ModeRun:
        texts_total = len(sample.texts)
        vectors = np.full((texts_total, 768), np.nan, dtype=np.float32)
        attempted_mask = np.ones((texts_total,), dtype=bool)
        success_mask = np.zeros((texts_total,), dtype=bool)
        per_item_wall_seconds = np.full((texts_total,), np.nan, dtype=np.float64)
        sent_token_counts = np.full((texts_total,), np.nan, dtype=np.float64)
        raw_shapes: list[str | None] = [None] * texts_total
        errors: list[str | None] = [None] * texts_total

        actual_batch_size = int(batch_size) if batch_size is not None else (16 if self.device == "cpu" else 128)
        actual_batch_size = max(1, int(actual_batch_size))
        started_at = perf_counter()
        with loop_progress(
            total=texts_total,
            label=f"local_{self.device} {sample.name}",
            enabled=bool(progress),
            unit="text",
        ) as tracker:
            for start in range(0, texts_total, actual_batch_size):
                stop = min(texts_total, start + actual_batch_size)
                batch_indices = np.arange(start, stop, dtype=np.int64)
                batch_texts = sample.texts[start:stop]
                batch_started_at = perf_counter()
                try:
                    enc = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=int(cap),
                    )
                    sent_lengths = enc["attention_mask"].sum(dim=1).detach().cpu().numpy().astype(np.int32, copy=False)
                    sent_token_counts[batch_indices] = sent_lengths.astype(np.float64, copy=False)
                    enc = {key: value.to(self.device) for key, value in enc.items()}
                    with self.torch.no_grad():
                        if self.device.startswith("cuda"):
                            self.torch.cuda.synchronize()
                        outputs = self.model(**enc)
                        if self.device.startswith("cuda"):
                            self.torch.cuda.synchronize()
                    cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
                    vectors[batch_indices] = cls
                    success_mask[batch_indices] = True
                    shape_repr = str(list(outputs.last_hidden_state.shape))
                    for idx in batch_indices:
                        raw_shapes[int(idx)] = shape_repr
                except Exception as exc:
                    for idx in batch_indices:
                        errors[int(idx)] = str(exc)
                batch_wall_seconds = perf_counter() - batch_started_at
                if len(batch_indices):
                    per_item_wall_seconds[batch_indices] = float(batch_wall_seconds / len(batch_indices))
                tracker.update(int(len(batch_indices)))

        total_wall_seconds = perf_counter() - started_at
        return _ModeRun(
            mode=f"local_{'gpu' if self.device.startswith('cuda') else 'cpu'}",
            available=True,
            vectors=vectors,
            attempted_mask=attempted_mask,
            success_mask=success_mask,
            per_item_wall_seconds=per_item_wall_seconds,
            raw_shapes=raw_shapes,
            sent_token_counts=sent_token_counts,
            errors=errors,
            load_seconds=float(self.load_seconds),
            processing_wall_seconds=float(total_wall_seconds),
            total_wall_seconds=float(self.load_seconds + total_wall_seconds),
            meta={
                "device": self.device,
                "batch_size": int(actual_batch_size),
                "cap": int(cap),
                "model_name": self.model_name,
            },
        )


def _try_create_local_session(*, model_name: str, device: str) -> tuple[_LocalSpecterSession | None, dict[str, Any]]:
    try:
        session = _LocalSpecterSession(model_name=model_name, device=device)
        return session, {"available": True, "device": device}
    except Exception as exc:
        return None, {"available": False, "device": device, "error": str(exc)}


def _build_track_sample_report(
    *,
    sample: _BenchmarkSample,
    cap: int,
    reference: _ModeRun,
    local_cpu: _ModeRun,
    hf_api: _ModeRun,
) -> dict[str, Any]:
    return {
        "sample_name": sample.name,
        "sample_size": int(len(sample.texts)),
        "sample_manifest": list(sample.manifest),
        "bucket_counts": _bucket_counts(sample.raw_token_counts, cap),
        "modes": {
            reference.mode: _summarize_mode_run(reference, sample, cap=cap),
            local_cpu.mode: _summarize_mode_run(
                local_cpu,
                sample,
                cap=cap,
                reference_vectors=reference.vectors,
                reference_success_mask=reference.success_mask,
            ),
            hf_api.mode: _summarize_mode_run(
                hf_api,
                sample,
                cap=cap,
                reference_vectors=reference.vectors,
                reference_success_mask=reference.success_mask,
            ),
        },
    }


def _run_track_b_downstream(
    *,
    output_root: Path,
    sample: _BenchmarkSample,
    local_gpu_vectors: np.ndarray,
    hf_vectors: np.ndarray,
    model_bundle: str | Path,
    dataset_id: str,
) -> dict[str, Any]:
    combined = sample.frame.reset_index(drop=True)
    pub_mask = combined["_benchmark_source"].eq("publications")
    ref_mask = combined["_benchmark_source"].eq("references")

    local_publications = _standardize_sample_frame(combined.loc[pub_mask].reset_index(drop=True), local_gpu_vectors[pub_mask.to_numpy()])
    hf_publications = _standardize_sample_frame(combined.loc[pub_mask].reset_index(drop=True), hf_vectors[pub_mask.to_numpy()])

    local_references = None
    hf_references = None
    if bool(ref_mask.any()):
        local_references = _standardize_sample_frame(combined.loc[ref_mask].reset_index(drop=True), local_gpu_vectors[ref_mask.to_numpy()])
        hf_references = _standardize_sample_frame(combined.loc[ref_mask].reset_index(drop=True), hf_vectors[ref_mask.to_numpy()])

    temp_root = Path(tempfile.mkdtemp(prefix="specter_benchmark_", dir=str(output_root)))
    try:
        local_root = temp_root / "sample_local"
        hf_root = temp_root / "sample_hf"
        local_root.mkdir(parents=True, exist_ok=True)
        hf_root.mkdir(parents=True, exist_ok=True)

        local_publications_path = local_root / "publications.parquet"
        hf_publications_path = hf_root / "publications.parquet"
        local_publications.to_parquet(local_publications_path, index=False)
        hf_publications.to_parquet(hf_publications_path, index=False)

        local_references_path = None
        hf_references_path = None
        if local_references is not None and hf_references is not None:
            local_references_path = local_root / "references.parquet"
            hf_references_path = hf_root / "references.parquet"
            local_references.to_parquet(local_references_path, index=False)
            hf_references.to_parquet(hf_references_path, index=False)

        smoke_local = run_infer_sources(
            InferSourcesRequest(
                publications_path=local_publications_path,
                references_path=local_references_path,
                output_root=temp_root / "infer_smoke_local",
                dataset_id=f"{dataset_id}__benchmark_local",
                model_bundle=model_bundle,
                infer_stage="smoke",
                device="cpu",
                precision_mode="fp32",
                cluster_backend="sklearn_cpu",
                force=True,
                progress=False,
            )
        )
        smoke_hf = run_infer_sources(
            InferSourcesRequest(
                publications_path=hf_publications_path,
                references_path=hf_references_path,
                output_root=temp_root / "infer_smoke_hf",
                dataset_id=f"{dataset_id}__benchmark_hf",
                model_bundle=model_bundle,
                infer_stage="smoke",
                device="cpu",
                precision_mode="fp32",
                cluster_backend="sklearn_cpu",
                force=True,
                progress=False,
            )
        )
        smoke_compare = _compare_infer_outputs(smoke_local, smoke_hf)

        mini_payload = None
        if smoke_compare["passed"]:
            mini_started_at = perf_counter()
            mini_result = run_infer_sources(
                InferSourcesRequest(
                    publications_path=hf_publications_path,
                    references_path=hf_references_path,
                    output_root=temp_root / "infer_mini_hf",
                    dataset_id=f"{dataset_id}__benchmark_hf_mini",
                    model_bundle=model_bundle,
                    infer_stage="mini",
                    device="cpu",
                    precision_mode="fp32",
                    cluster_backend="sklearn_cpu",
                    force=True,
                    progress=False,
                )
            )
            mini_wall_seconds = float(perf_counter() - mini_started_at)
            mini_stage_metrics = {}
            if mini_result.stage_metrics_path.exists():
                mini_stage_metrics = json.loads(mini_result.stage_metrics_path.read_text(encoding="utf-8"))
            mini_clusters = pd.read_parquet(mini_result.mention_clusters_path)
            mini_payload = {
                "wall_seconds": float(mini_wall_seconds),
                "go": mini_result.go,
                "mention_count": int(len(mini_clusters)),
                "cluster_count": int(mini_clusters["author_uid"].nunique()) if len(mini_clusters) else 0,
                "stage_metrics": mini_stage_metrics,
            }
        return {
            "smoke": smoke_compare,
            "mini": mini_payload,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _compute_full_ads_scaling_stats(
    *,
    publications_path: str | Path,
    references_path: str | Path | None,
) -> dict[str, Any]:
    mentions = normalize_ads_mentions(
        publications_path=publications_path,
        references_path=references_path,
    )
    full_cfg = _resolve_infer_run_cfg("full")
    pair_upper_bound = _estimate_pair_upper_bound(
        mentions,
        max_pairs_per_block=full_cfg.get("infer_overrides", {}).get("max_pairs_per_block"),
    )
    return {
        "mentions_total": int(len(mentions)),
        "blocks_total": int(mentions["block_key"].nunique()) if len(mentions) else 0,
        "pair_upper_bound": int(pair_upper_bound),
    }


def _embedding_extrapolation(mode_summary: dict[str, Any], total_sources: int) -> dict[str, Any]:
    texts_per_second = mode_summary.get("texts_per_second")
    per_text = mode_summary.get("per_text_wall_seconds", {}).get("mean")
    if not isinstance(texts_per_second, (int, float)) or float(texts_per_second) <= 0:
        return {"full_embed_seconds": None}
    return {
        "full_embed_seconds": float(total_sources / float(texts_per_second)),
        "mean_seconds_per_text": None if per_text is None else float(per_text),
    }


def _build_tail_extrapolation(
    *,
    downstream_payload: dict[str, Any] | None,
    full_source_rows: int,
    full_mentions: int,
    full_pairs: int,
) -> dict[str, Any] | None:
    if not downstream_payload:
        return None
    mini = downstream_payload.get("mini")
    if not isinstance(mini, dict) or not mini:
        return None
    stage_metrics = dict(mini.get("stage_metrics") or {})
    runtime = dict(stage_metrics.get("runtime") or {})
    counts = dict(stage_metrics.get("counts") or {})
    load_inputs = dict(runtime.get("load_inputs") or {})
    pair_building = dict(runtime.get("pair_building") or {})
    sample_source_rows = int(load_inputs.get("input_record_count") or 0)
    sample_mentions = int(counts.get("ads_mentions") or mini.get("mention_count") or 0)
    sample_pairs = int(pair_building.get("pairs_written") or pair_building.get("total_pairs_est") or 0)
    if sample_source_rows <= 0 or sample_mentions <= 0 or sample_pairs <= 0:
        return None
    source_ratio = float(full_source_rows / sample_source_rows)
    mention_ratio = float(full_mentions / sample_mentions)
    pair_ratio = float(full_pairs / sample_pairs)
    chosen_tail_seconds = float(mini["wall_seconds"]) * pair_ratio
    return {
        "sample_wall_seconds": float(mini["wall_seconds"]),
        "sample_source_rows": int(sample_source_rows),
        "sample_mentions": int(sample_mentions),
        "sample_pairs": int(sample_pairs),
        "ratios": {
            "source_ratio": float(source_ratio),
            "mention_ratio": float(mention_ratio),
            "pair_ratio": float(pair_ratio),
        },
        "source_scaled_seconds": float(mini["wall_seconds"]) * source_ratio,
        "mention_scaled_seconds": float(mini["wall_seconds"]) * mention_ratio,
        "pair_scaled_seconds": float(chosen_tail_seconds),
        "chosen_tail_seconds": float(chosen_tail_seconds),
        "chosen_method": "pair_scaled",
    }


def _write_markdown_report(payload: dict[str, Any], path: Path) -> Path:
    track_a = payload["tracks"]["track_a"]
    track_b = payload["tracks"]["track_b"]
    throughput_a = track_a["throughput_sample"]["modes"]
    throughput_b = track_b["throughput_sample"]["modes"]
    decision = payload["decision"]
    raw_probe = payload["raw_hf_probe"]["mode"]
    lines = [
        "# SPECTER Benchmark Report",
        "",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Bundle: `{payload['model_bundle']}`",
        f"- Recommendation: `{decision['recommendation']}`",
        f"- Raw HF API feasible on long ADS texts: `{decision['hf_api_raw_long_text_feasible']}`",
        f"- HF API with client-side truncation viable for Track B: `{decision['hf_api_truncated_track_b_viable']}`",
        "",
        "## Why The Notebook MWE Works",
        "",
        f"- Notebook MWE tokens: `{payload['mwe_sanity']['raw_token_count']}`",
        f"- Track A cap: `{track_a['cap']}`",
        f"- Track B cap: `{track_b['cap']}`",
        "",
        "## Full Input Stats",
        "",
        f"- Total source rows: `{payload['full_dataset']['source_rows_total']}`",
        f"- Publications: `{payload['full_dataset']['publications_rows']}`",
        f"- References: `{payload['full_dataset']['references_rows']}`",
        f"- Raw token max: `{payload['full_dataset']['raw_token_lengths']['max']}`",
        f"- `>256` share: `{payload['full_dataset']['bucket_counts_track_b']['> 512']}` in `>512` and `{payload['full_dataset']['bucket_counts_track_b']['257-512']}` in `257-512`",
        "",
        "## Track A Throughput",
        "",
        f"- local_gpu texts/sec: `{throughput_a['local_gpu']['texts_per_second']}`",
        f"- local_cpu texts/sec: `{throughput_a['local_cpu']['texts_per_second']}`",
        f"- hf_api_truncated texts/sec: `{throughput_a['hf_api_truncated']['texts_per_second']}`",
        f"- hf_api_truncated api_concurrency: `{throughput_a['hf_api_truncated']['meta']['api_concurrency']}`",
        "",
        "## Track B Throughput",
        "",
        f"- local_gpu texts/sec: `{throughput_b['local_gpu']['texts_per_second']}`",
        f"- local_cpu texts/sec: `{throughput_b['local_cpu']['texts_per_second']}`",
        f"- hf_api_truncated texts/sec: `{throughput_b['hf_api_truncated']['texts_per_second']}`",
        f"- hf_api_truncated cosine mean vs local_gpu: `{throughput_b['hf_api_truncated']['cosine_vs_local_gpu']['mean_cosine_similarity']}`",
        "",
        "## Raw HF Probe",
        "",
        f"- Success/full on parity sample: `{raw_probe['texts_successful']}` / `{raw_probe['texts_total']}`",
        f"- Failed texts on parity sample: `{raw_probe['texts_failed']}`",
    ]
    downstream = track_b.get("downstream")
    if downstream:
        lines.extend(
            [
                "",
                "## Track B Downstream",
                "",
                f"- Smoke passed: `{downstream['smoke']['passed']}`",
                f"- Changed assignments: `{downstream['smoke']['changed_assignments']}`",
                f"- Mini run available: `{bool(downstream.get('mini'))}`",
            ]
        )
        mini = downstream.get("mini")
        if mini:
            lines.extend(
                [
                    f"- Mini CPU wall seconds: `{mini['wall_seconds']}`",
                    f"- Mini mentions: `{mini['mention_count']}`",
                    f"- Mini clusters: `{mini['cluster_count']}`",
                ]
            )
    extrap = payload["extrapolation"]
    lines.extend(
        [
            "",
            "## Full-Run Estimates",
            "",
            f"- Track A local_gpu embed-only full seconds: `{extrap['track_a']['embedding_only']['local_gpu']['full_embed_seconds']}`",
            f"- Track B local_cpu embed-only full seconds: `{extrap['track_b']['embedding_only']['local_cpu']['full_embed_seconds']}`",
            f"- Track B hf_api_truncated embed-only full seconds: `{extrap['track_b']['embedding_only']['hf_api_truncated']['full_embed_seconds']}`",
        ]
    )
    tail = extrap["track_b"].get("cpu_infer_tail")
    if tail:
        lines.extend(
            [
                f"- Track B chosen CPU tail seconds: `{tail['chosen_tail_seconds']}`",
                f"- Track B hf_api_truncated end-to-end full seconds: `{extrap['track_b']['end_to_end']['hf_api_truncated_full_seconds']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The notebook MWE succeeds because it is a short paper text and stays below the problematic raw long-text regime.",
            "- The main comparison is cap-aligned with the live inference path: local GPU, local CPU, and HF API all use the same tokenizer truncation per track.",
            "- The raw HF endpoint is kept only as a small diagnostic probe so long-text provider failures stay visible without dominating the benchmark.",
            "- Track A is the notebook/SPECTER view; Track B is the actual bundle view.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_specter_benchmark(request: SpecterBenchmarkRequest) -> SpecterBenchmarkResult:
    run_id = default_run_id("specter_benchmark", tag="cli")
    output_root = _resolved_path(request.output_root)
    report_json_path = output_root / "specter_benchmark_report.json"
    report_markdown_path = output_root / "specter_benchmark_report.md"
    if output_root.exists() and not request.force and (report_json_path.exists() or report_markdown_path.exists()):
        raise FileExistsError(
            "Refusing to overwrite existing benchmark artifacts without force=True: "
            f"{report_json_path}, {report_markdown_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    model_info = _resolve_model_bundle(request.model_bundle)
    bundle_contract = dict(
        model_info["manifest"].get("embedding_contract") or build_bundle_embedding_contract(model_info["model_cfg"])
    )
    bundle_text_contract = dict(bundle_contract.get("text", {}) or {})
    model_name = _normalize_model_name(request.model_name)
    if bundle_text_contract.get("model_name") and str(bundle_text_contract["model_name"]) != model_name:
        raise ValueError(
            "SPECTER benchmark must use the bundle text embedding contract. "
            f"bundle={bundle_text_contract.get('model_name')!r} request={model_name!r}"
        )

    track_b_cap = int(bundle_text_contract.get("tokenization", {}).get("max_length", _DEFAULT_TRACK_B_CAP))
    tokenizer = _build_tokenizer(model_name)
    hardware = _collect_hardware_metadata()

    publications = _load_normalized_source(request.publications_path, source_type="publication")
    references = (
        None if request.references_path is None else _load_normalized_source(request.references_path, source_type="reference")
    )
    combined = _build_combined_sources(publications, references)
    if len(combined) == 0:
        raise RuntimeError("Benchmark input is empty; provide at least one source record.")

    raw_token_counts_full = _compute_raw_token_counts_for_frame(combined, tokenizer=tokenizer)
    full_token_stats = _summarize_token_lengths(raw_token_counts_full)

    parity_sample = _make_sample(
        combined,
        name="parity_sample",
        target_size=int(request.parity_sample_size),
        raw_token_counts_full=raw_token_counts_full,
    )
    throughput_sample = _make_sample(
        combined,
        name="throughput_sample",
        target_size=int(request.throughput_sample_size),
        raw_token_counts_full=raw_token_counts_full,
    )

    notebook_mwe = _load_notebook_mwe_text()
    mwe_text = str(notebook_mwe["text"])
    mwe_counts = _compute_raw_token_counts_for_texts([mwe_text], tokenizer=tokenizer)
    mwe_sample = _BenchmarkSample(
        name="mwe_sanity",
        frame=pd.DataFrame(
            [
                {
                    "bibcode": "mwe",
                    "authors": [],
                    "title": notebook_mwe["title"],
                    "abstract": notebook_mwe["abstract"],
                    "year": None,
                    "aff": [],
                    "_benchmark_source": "mwe",
                    "_benchmark_source_order": 0,
                }
            ]
        ),
        texts=[mwe_text],
        raw_token_counts=mwe_counts.astype(np.int32, copy=False),
        manifest=[
            {
                "sample_index": 0,
                "source": "mwe",
                "source_order": 0,
                "bibcode": "mwe",
                "year": None,
                "raw_token_count": int(mwe_counts[0]),
            }
        ],
    )

    local_cpu_session, local_cpu_meta = _try_create_local_session(model_name=model_name, device=request.cpu_device)
    local_gpu_session, local_gpu_meta = _try_create_local_session(model_name=model_name, device=request.gpu_device)
    if local_gpu_session is None:
        raise RuntimeError(f"Benchmark requires a working GPU session for local reference: {local_gpu_meta.get('error')}")
    if local_cpu_session is None:
        raise RuntimeError(f"Benchmark requires a working CPU session: {local_cpu_meta.get('error')}")

    hf_raw_probe = _run_hf_mode(
        sample=parity_sample,
        mode_name="hf_api_raw_probe",
        model_name=model_name,
        provider=request.provider,
        hf_token_env_var=request.hf_token_env_var,
        tokenizer=tokenizer,
        progress=bool(request.progress),
    )
    raw_long_text_failure = any(
        (not bool(hf_raw_probe.success_mask[idx])) and int(parity_sample.raw_token_counts[idx]) > 512
        for idx in range(len(parity_sample.texts))
        if bool(hf_raw_probe.attempted_mask[idx])
    )

    def _run_track(cap: int, *, sample: _BenchmarkSample) -> tuple[dict[str, Any], _ModeRun, _ModeRun, _ModeRun]:
        local_gpu = local_gpu_session.run(
            sample=sample,
            cap=int(cap),
            batch_size=request.local_batch_size,
            progress=bool(request.progress),
        )
        local_cpu = local_cpu_session.run(
            sample=sample,
            cap=int(cap),
            batch_size=request.local_batch_size,
            progress=bool(request.progress),
        )
        hf_api = _run_hf_mode(
            sample=sample,
            mode_name="hf_api_truncated",
            model_name=model_name,
            provider=request.provider,
            hf_token_env_var=request.hf_token_env_var,
            tokenizer=tokenizer,
            cap=int(cap),
            progress=bool(request.progress),
            api_concurrency=int(request.api_concurrency),
        )
        report = _build_track_sample_report(
            sample=sample,
            cap=int(cap),
            reference=local_gpu,
            local_cpu=local_cpu,
            hf_api=hf_api,
        )
        return report, local_gpu, local_cpu, hf_api

    mwe_local_gpu = local_gpu_session.run(sample=mwe_sample, cap=_TRACK_A_CAP, batch_size=1, progress=False)
    mwe_local_cpu = local_cpu_session.run(sample=mwe_sample, cap=_TRACK_A_CAP, batch_size=1, progress=False)
    mwe_hf_raw = _run_hf_mode(
        sample=mwe_sample,
        mode_name="hf_api_raw_probe",
        model_name=model_name,
        provider=request.provider,
        hf_token_env_var=request.hf_token_env_var,
        tokenizer=tokenizer,
        progress=False,
    )
    mwe_hf_truncated = _run_hf_mode(
        sample=mwe_sample,
        mode_name="hf_api_truncated",
        model_name=model_name,
        provider=request.provider,
        hf_token_env_var=request.hf_token_env_var,
        tokenizer=tokenizer,
        cap=_TRACK_A_CAP,
        progress=False,
        api_concurrency=int(request.api_concurrency),
    )

    track_a_parity, _, _, _ = _run_track(_TRACK_A_CAP, sample=parity_sample)
    track_a_throughput, _, _, _ = _run_track(_TRACK_A_CAP, sample=throughput_sample)
    track_b_parity, _, _, _ = _run_track(track_b_cap, sample=parity_sample)
    track_b_throughput, track_b_throughput_local_gpu, _, track_b_throughput_hf_api = _run_track(
        track_b_cap,
        sample=throughput_sample,
    )

    downstream = None
    track_b_candidate_summary = track_b_throughput["modes"]["hf_api_truncated"]
    if (
        isinstance(track_b_candidate_summary.get("cosine_vs_local_gpu"), dict)
        and bool(track_b_candidate_summary["texts_successful"] == track_b_candidate_summary["texts_total"])
    ):
        downstream = _run_track_b_downstream(
            output_root=output_root,
            sample=throughput_sample,
            local_gpu_vectors=track_b_throughput_local_gpu.vectors,
            hf_vectors=track_b_throughput_hf_api.vectors,
            model_bundle=request.model_bundle,
            dataset_id=request.dataset_id,
        )

    full_ads_scaling = _compute_full_ads_scaling_stats(
        publications_path=request.publications_path,
        references_path=request.references_path,
    )
    full_source_rows = int(len(combined))

    track_a_embedding_only = {
        "local_gpu": _embedding_extrapolation(track_a_throughput["modes"]["local_gpu"], full_source_rows),
        "local_cpu": _embedding_extrapolation(track_a_throughput["modes"]["local_cpu"], full_source_rows),
        "hf_api_truncated": _embedding_extrapolation(
            track_a_throughput["modes"]["hf_api_truncated"],
            full_source_rows,
        ),
    }
    track_b_embedding_only = {
        "local_gpu": _embedding_extrapolation(track_b_throughput["modes"]["local_gpu"], full_source_rows),
        "local_cpu": _embedding_extrapolation(track_b_throughput["modes"]["local_cpu"], full_source_rows),
        "hf_api_truncated": _embedding_extrapolation(
            track_b_throughput["modes"]["hf_api_truncated"],
            full_source_rows,
        ),
    }
    cpu_tail = _build_tail_extrapolation(
        downstream_payload=downstream,
        full_source_rows=full_source_rows,
        full_mentions=int(full_ads_scaling["mentions_total"]),
        full_pairs=int(full_ads_scaling["pair_upper_bound"]),
    )
    end_to_end = {
        "hf_api_truncated_full_seconds": None,
        "local_cpu_full_seconds": None,
    }
    if cpu_tail is not None:
        hf_embed = track_b_embedding_only["hf_api_truncated"].get("full_embed_seconds")
        cpu_embed = track_b_embedding_only["local_cpu"].get("full_embed_seconds")
        if isinstance(hf_embed, (int, float)):
            end_to_end["hf_api_truncated_full_seconds"] = float(hf_embed + cpu_tail["chosen_tail_seconds"])
        if isinstance(cpu_embed, (int, float)):
            end_to_end["local_cpu_full_seconds"] = float(cpu_embed + cpu_tail["chosen_tail_seconds"])

    recommendation = (
        "Track B benchmark passes with pooled HF API after client-side tokenizer truncation."
        if downstream and downstream.get("smoke", {}).get("passed")
        else "HF stays experimental unless the cap-aligned truncated API path passes Track B downstream parity."
    )
    payload = {
        "run_id": run_id,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset_id": str(request.dataset_id),
        "model_bundle": str(_resolved_path(request.model_bundle)),
        "embedding_contract": bundle_contract,
        "hardware": hardware,
        "mwe_sanity": {
            "notebook_path": notebook_mwe["notebook_path"],
            "raw_token_count": int(mwe_counts[0]),
            "modes": {
                "local_gpu": _summarize_mode_run(mwe_local_gpu, mwe_sample, cap=_TRACK_A_CAP),
                "local_cpu": _summarize_mode_run(
                    mwe_local_cpu,
                    mwe_sample,
                    cap=_TRACK_A_CAP,
                    reference_vectors=mwe_local_gpu.vectors,
                    reference_success_mask=mwe_local_gpu.success_mask,
                ),
                "hf_api_raw": _summarize_mode_run(
                    mwe_hf_raw,
                    mwe_sample,
                    cap=_TRACK_A_CAP,
                    reference_vectors=mwe_local_gpu.vectors,
                    reference_success_mask=mwe_local_gpu.success_mask,
                ),
                "hf_api_truncated": _summarize_mode_run(
                    mwe_hf_truncated,
                    mwe_sample,
                    cap=_TRACK_A_CAP,
                    reference_vectors=mwe_local_gpu.vectors,
                    reference_success_mask=mwe_local_gpu.success_mask,
                ),
            },
        },
        "raw_hf_probe": {
            "sample_name": parity_sample.name,
            "sample_size": int(len(parity_sample.texts)),
            "bucket_counts_track_a": _bucket_counts(parity_sample.raw_token_counts, _TRACK_A_CAP),
            "bucket_counts_track_b": _bucket_counts(parity_sample.raw_token_counts, track_b_cap),
            "mode": _summarize_mode_run(hf_raw_probe, parity_sample, cap=track_b_cap),
        },
        "full_dataset": {
            "publications_rows": int(len(publications)),
            "references_rows": 0 if references is None else int(len(references)),
            "source_rows_total": int(full_source_rows),
            "raw_token_lengths": full_token_stats,
            "bucket_counts_track_a": _bucket_counts(raw_token_counts_full, _TRACK_A_CAP),
            "bucket_counts_track_b": _bucket_counts(raw_token_counts_full, track_b_cap),
            "ads_scaling": full_ads_scaling,
        },
        "tracks": {
            "track_a": {
                "cap": int(_TRACK_A_CAP),
                "parity_sample": track_a_parity,
                "throughput_sample": track_a_throughput,
            },
            "track_b": {
                "cap": int(track_b_cap),
                "parity_sample": track_b_parity,
                "throughput_sample": track_b_throughput,
                "downstream": downstream,
            },
        },
        "extrapolation": {
            "track_a": {
                "embedding_only": track_a_embedding_only,
            },
            "track_b": {
                "embedding_only": track_b_embedding_only,
                "cpu_infer_tail": cpu_tail,
                "end_to_end": end_to_end,
            },
        },
        "decision": {
            "hf_api_raw_long_text_feasible": not bool(raw_long_text_failure),
            "hf_api_truncated_track_b_viable": bool(downstream and downstream.get("smoke", {}).get("passed")),
            "recommendation": recommendation,
        },
        "artifacts": {
            "output_root": str(output_root),
            "report_json_path": str(report_json_path),
            "report_markdown_path": str(report_markdown_path),
        },
    }
    write_json(payload, report_json_path)
    _write_markdown_report(payload, report_markdown_path)
    return SpecterBenchmarkResult(
        run_id=run_id,
        output_root=output_root,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        recommendation=recommendation,
    )
