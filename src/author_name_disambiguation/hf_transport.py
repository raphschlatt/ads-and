from __future__ import annotations

import asyncio
import importlib.util
import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from author_name_disambiguation.common.cli_ui import loop_progress
from author_name_disambiguation.embedding_contract import DEFAULT_TEXT_MODEL_NAME, TEXT_EMBEDDING_DIM
from author_name_disambiguation.features.specter_runtime import compute_token_length_order, load_tokenizer_prefer_fast

DEFAULT_HF_PROVIDER = "hf-inference"
DEFAULT_HF_TOKEN_ENV_VAR = "HF_TOKEN"
DEFAULT_HF_CONCURRENCY = 16
DEFAULT_HF_MAX_RETRIES = 5
DEFAULT_HF_BASE_BACKOFF_SECONDS = 1.0
DEFAULT_HF_MAX_BACKOFF_SECONDS = 30.0
DEFAULT_HF_CHUNK_SIZE = 64
DEFAULT_HF_DEGRADE_STEPS: tuple[tuple[bool, int], ...] = (
    (True, 16),
    (False, 16),
    (False, 8),
    (False, 4),
)


@dataclass(frozen=True, slots=True)
class HFTransportProfile:
    name: str
    http2_requested: bool
    concurrency: int
    verify: bool = True


def normalize_provider(provider: str) -> str:
    value = str(provider or DEFAULT_HF_PROVIDER).strip().lower()
    if value != DEFAULT_HF_PROVIDER:
        raise ValueError(f"Unsupported provider={provider!r}. Expected {DEFAULT_HF_PROVIDER!r}.")
    return value


def normalize_model_name(model_name: str) -> str:
    value = str(model_name or DEFAULT_TEXT_MODEL_NAME).strip()
    if value != DEFAULT_TEXT_MODEL_NAME:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Expected {DEFAULT_TEXT_MODEL_NAME!r}."
        )
    return value


def resolve_hf_token(env_var_name: str = DEFAULT_HF_TOKEN_ENV_VAR) -> str:
    key = str(env_var_name or DEFAULT_HF_TOKEN_ENV_VAR).strip() or DEFAULT_HF_TOKEN_ENV_VAR
    token = os.environ.get(key)
    if token and token.strip():
        return token.strip()
    raise RuntimeError(
        f"Missing Hugging Face token in environment variable {key!r}. "
        "Export HF_TOKEN before running remote SPECTER inference."
    )


def is_retryable_hf_error(exc: BaseException) -> bool:
    status_candidates: list[int] = []
    for attr in ("status_code", "response"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            status_candidates.append(int(value))
        elif value is not None:
            status = getattr(value, "status_code", None)
            if isinstance(status, int):
                status_candidates.append(int(status))

    if any(code == 429 or 500 <= code < 600 for code in status_candidates):
        return True

    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "503" in text or "502" in text or "500" in text


def normalize_single_embedding(item: Any, *, dim: int = TEXT_EMBEDDING_DIM) -> np.ndarray:
    try:
        arr = np.asarray(item, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"HF response item could not be converted to float32: {type(item).__name__}") from exc

    if arr.ndim == 1 and arr.shape[0] == dim:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[1] == dim and arr.shape[0] >= 1:
        return arr[0].astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == dim and arr.shape[1] >= 1:
        return arr[0, 0].astype(np.float32, copy=False)
    raise ValueError(f"Incompatible HF response item shape {tuple(arr.shape)}; expected 768-d doc or token features.")


def normalize_hf_batch_response(
    payload: Any,
    *,
    expected_items: int,
    dim: int = TEXT_EMBEDDING_DIM,
) -> np.ndarray:
    if expected_items <= 0:
        return np.zeros((0, dim), dtype=np.float32)

    if expected_items == 1:
        return normalize_single_embedding(payload, dim=dim).reshape(1, dim)

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if len(payload) == expected_items:
            rows = [normalize_single_embedding(item, dim=dim) for item in payload]
            return np.vstack(rows).astype(np.float32, copy=False)

    try:
        arr = np.asarray(payload, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError("HF response could not be normalized to float32 arrays.") from exc

    if arr.ndim == 2 and arr.shape == (expected_items, dim):
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[0] == expected_items and arr.shape[2] == dim:
        return arr[:, 0, :].astype(np.float32, copy=False)
    raise ValueError(
        "Incompatible HF batch response shape "
        f"{tuple(arr.shape)} for expected_items={expected_items}; expected ({expected_items}, 768) "
        f"or token-level ({expected_items}, seq_len, 768)."
    )


def hf_http2_enabled() -> bool:
    return importlib.util.find_spec("h2") is not None


def build_hf_feature_extraction_url(*, provider: str, model_name: str) -> str:
    from huggingface_hub import constants

    base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider=provider)
    return f"{base_url}/models/{model_name}/pipeline/feature-extraction"


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


def _batched(items: Sequence[int], batch_size: int) -> Iterable[list[int]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _import_httpx():
    try:
        import httpx

        return httpx
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "HF runtime mode requires product dependency `httpx`. "
            "Reinstall the package with current project dependencies."
        ) from exc


def _summarize_hf_payload_shape(payload: Any) -> list[int] | str | None:
    try:
        arr = np.asarray(payload, dtype=np.float32)
    except Exception:
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            outer = int(len(payload))
            inner_kinds: list[str] = []
            for item in payload[:3]:
                try:
                    inner = np.asarray(item, dtype=np.float32)
                    inner_kinds.append(str(tuple(int(v) for v in inner.shape)))
                except Exception:
                    inner_kinds.append(type(item).__name__)
            inner_summary = ",".join(inner_kinds) if inner_kinds else "empty"
            suffix = ",..." if outer > 3 else ""
            return f"ragged[{outer}]<{inner_summary}{suffix}>"
        return None
    return [int(v) for v in arr.shape]


async def _request_hf_batch_async(
    *,
    client: Any,
    url: str,
    texts: Sequence[str],
    max_retries: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    httpx = _import_httpx()
    texts_batch = [str(text) for text in texts]
    if len(texts_batch) == 0:
        return np.zeros((0, TEXT_EMBEDDING_DIM), dtype=np.float32), {
            "attempts": 0,
            "backoff_seconds_total": 0.0,
            "wall_seconds": 0.0,
            "raw_shape": [0, TEXT_EMBEDDING_DIM],
            "batch_size": 0,
        }

    started_at = perf_counter()
    attempts = 0
    backoff_seconds_total = 0.0
    request_payload = {
        "inputs": texts_batch[0] if len(texts_batch) == 1 else texts_batch,
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
            vectors = normalize_hf_batch_response(payload, expected_items=len(texts_batch))
            return vectors.astype(np.float32, copy=False), {
                "attempts": int(attempts),
                "backoff_seconds_total": float(backoff_seconds_total),
                "wall_seconds": float(perf_counter() - started_at),
                "raw_shape": _summarize_hf_payload_shape(payload),
                "batch_size": int(len(texts_batch)),
            }
        except Exception as exc:
            should_retry = attempts <= int(max_retries) and is_retryable_hf_error(exc)
            if not should_retry:
                detail = str(exc)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    body = exc.response.text.strip()
                    if body:
                        detail = f"HTTP {exc.response.status_code}: {body[:240]}"
                raise RuntimeError(
                    f"HF feature_extraction batch failed after {attempts} attempt(s) "
                    f"for {len(texts_batch)} text(s): {type(exc).__name__}: {detail}"
                ) from exc
            delay = min(float(max_backoff_seconds), float(base_backoff_seconds) * (2 ** (attempts - 1)))
            await asyncio.sleep(delay)
            backoff_seconds_total += float(delay)


def _prepare_truncated_texts(
    *,
    texts: Sequence[str],
    model_name: str,
    max_length: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if len(texts) == 0:
        return [], np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    tokenizer = load_tokenizer_prefer_fast(model_name)
    text_list = [str(text) for text in texts]
    order = compute_token_length_order(text_list, tokenizer=tokenizer, max_length=max_length)
    ordered_texts: list[str] = []
    ordered_token_counts: list[int] = []
    for idx in order.tolist():
        truncated_text, _truncated_tokens, retokenized_tokens = _truncate_text_to_cap(
            text_list[int(idx)],
            tokenizer=tokenizer,
            cap=int(max_length),
        )
        ordered_texts.append(truncated_text)
        ordered_token_counts.append(int(retokenized_tokens))
    return ordered_texts, order.astype(np.int64), np.asarray(ordered_token_counts, dtype=np.int64)


def _default_profiles() -> list[HFTransportProfile]:
    profiles: list[HFTransportProfile] = []
    for http2_requested, concurrency in DEFAULT_HF_DEGRADE_STEPS:
        profiles.append(
            HFTransportProfile(
                name=f"httpx_hf_http{'2' if http2_requested else '1'}_c{int(concurrency)}",
                http2_requested=bool(http2_requested),
                concurrency=int(concurrency),
                verify=True,
            )
        )
    return profiles


def _select_profiles(*, concurrency: int | None, use_http2: bool | None) -> list[HFTransportProfile]:
    if concurrency is None and use_http2 is None:
        return _default_profiles()
    selected_concurrency = DEFAULT_HF_CONCURRENCY if concurrency is None else max(1, int(concurrency))
    selected_http2 = hf_http2_enabled() if use_http2 is None else bool(use_http2)
    return [
        HFTransportProfile(
            name=f"httpx_hf_http{'2' if selected_http2 else '1'}_c{selected_concurrency}",
            http2_requested=bool(selected_http2),
            concurrency=int(selected_concurrency),
            verify=True,
        )
    ]


def _summarize_errors(errors: Sequence[str | None], *, limit: int = 5) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for error in errors:
        if not error:
            continue
        counts[str(error)] = counts.get(str(error), 0) + 1
    rows = [{"value": key, "count": int(value)} for key, value in counts.items()]
    rows.sort(key=lambda row: (-int(row["count"]), str(row["value"])))
    return rows[:limit]


def _run_httpx_profile(
    *,
    texts: Sequence[str],
    token_counts: np.ndarray,
    provider: str,
    model_name: str,
    api_key: str,
    profile: HFTransportProfile,
    chunk_size: int,
    max_retries: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    progress: bool,
    progress_label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    httpx = _import_httpx()
    texts_total = len(texts)
    vectors = np.full((texts_total, TEXT_EMBEDDING_DIM), np.nan, dtype=np.float32)
    success_mask = np.zeros((texts_total,), dtype=bool)
    errors: list[str | None] = [None] * texts_total
    raw_shapes: list[str | None] = [None] * texts_total
    per_item_wall_seconds = np.full((texts_total,), np.nan, dtype=np.float64)
    attempts_total = 0
    backoff_seconds_total = 0.0
    request_url = build_hf_feature_extraction_url(provider=provider, model_name=model_name)
    effective_http2 = bool(profile.http2_requested and hf_http2_enabled())
    request_batch_size = max(1, int(chunk_size))
    text_batches = _batched(list(range(texts_total)), batch_size=request_batch_size)
    batch_indices_list = [batch for batch in text_batches if batch]

    async def _run_all(tracker: Any) -> None:
        nonlocal attempts_total, backoff_seconds_total
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
        limits = httpx.Limits(
            max_connections=int(profile.concurrency),
            max_keepalive_connections=int(profile.concurrency),
        )
        headers = {"Authorization": f"Bearer {api_key}"}
        semaphore = asyncio.Semaphore(int(profile.concurrency))

        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=limits,
            http2=bool(effective_http2),
            verify=bool(profile.verify),
        ) as client:

            async def _run_batch(batch_indices: list[int]) -> None:
                nonlocal attempts_total, backoff_seconds_total
                try:
                    async with semaphore:
                        batch_vectors, meta = await _request_hf_batch_async(
                            client=client,
                            url=request_url,
                            texts=[str(texts[idx]) for idx in batch_indices],
                            max_retries=max_retries,
                            base_backoff_seconds=base_backoff_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                        )
                    for offset, idx in enumerate(batch_indices):
                        vectors[idx] = batch_vectors[offset]
                        success_mask[idx] = True
                        per_item_wall_seconds[idx] = float(meta.get("wall_seconds") or 0.0) / max(
                            1, len(batch_indices)
                        )
                        raw_shapes[idx] = None if meta.get("raw_shape") is None else str(meta["raw_shape"])
                    attempts_total += int(meta.get("attempts") or 0)
                    backoff_seconds_total += float(meta.get("backoff_seconds_total") or 0.0)
                except Exception as exc:
                    for idx in batch_indices:
                        errors[idx] = str(exc)
                finally:
                    tracker.update(len(batch_indices))

            for launch_group in _batched(batch_indices_list, batch_size=max(1, int(profile.concurrency))):
                await asyncio.gather(*(_run_batch(batch_indices) for batch_indices in launch_group))

    started_at = perf_counter()
    with loop_progress(
        total=texts_total,
        label=progress_label,
        enabled=bool(progress),
        unit="text",
    ) as tracker:
        asyncio.run(_run_all(tracker))

    failed_count = int(texts_total - int(success_mask.sum()))
    if failed_count > 0:
        raise RuntimeError(
            f"HF HTTPX profile {profile.name!r} failed for {failed_count}/{texts_total} texts. "
            f"Top errors: {_summarize_errors(errors)!r}"
        )

    return vectors, {
        "transport": "httpx_async_pool",
        "profile_name": profile.name,
        "provider": provider,
        "model_name": model_name,
        "api_concurrency": int(profile.concurrency),
        "http2_requested": bool(profile.http2_requested),
        "http2_enabled": bool(effective_http2),
        "verify": bool(profile.verify),
        "request_batch_size": int(request_batch_size),
        "texts_total": int(texts_total),
        "texts_successful": int(success_mask.sum()),
        "texts_failed": int(failed_count),
        "attempts_total": int(attempts_total),
        "backoff_seconds_total": float(backoff_seconds_total),
        "processing_wall_seconds": float(perf_counter() - started_at),
        "mean_sent_token_count": None
        if token_counts.size == 0
        else float(np.mean(token_counts.astype(np.float64))),
        "max_sent_token_count": None if token_counts.size == 0 else int(np.max(token_counts)),
        "raw_shapes_top": _summarize_errors(raw_shapes),
        "per_item_wall_seconds_mean": None
        if not np.isfinite(per_item_wall_seconds).any()
        else float(np.nanmean(per_item_wall_seconds)),
    }


def embed_texts_via_hf_httpx(
    *,
    texts: Sequence[str],
    provider: str = DEFAULT_HF_PROVIDER,
    model_name: str = DEFAULT_TEXT_MODEL_NAME,
    hf_token_env_var: str = DEFAULT_HF_TOKEN_ENV_VAR,
    max_length: int = 256,
    concurrency: int | None = None,
    use_http2: bool | None = None,
    chunk_size: int = DEFAULT_HF_CHUNK_SIZE,
    max_retries: int = DEFAULT_HF_MAX_RETRIES,
    base_backoff_seconds: float = DEFAULT_HF_BASE_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_HF_MAX_BACKOFF_SECONDS,
    progress: bool = False,
    progress_label: str = "HF remote SPECTER",
) -> tuple[np.ndarray, dict[str, Any]]:
    provider = normalize_provider(provider)
    model_name = normalize_model_name(model_name)
    api_key = resolve_hf_token(hf_token_env_var)
    started_at = perf_counter()
    prepared_texts, order, ordered_token_counts = _prepare_truncated_texts(
        texts=texts,
        model_name=model_name,
        max_length=max_length,
    )
    if len(prepared_texts) == 0:
        return np.zeros((0, TEXT_EMBEDDING_DIM), dtype=np.float32), {
            "transport": "httpx_async_pool",
            "profile_name": None,
            "provider": provider,
            "model_name": model_name,
            "api_concurrency": None,
            "http2_requested": None,
            "http2_enabled": None,
            "verify": True,
            "texts_total": 0,
            "texts_successful": 0,
            "texts_failed": 0,
            "attempts_total": 0,
            "backoff_seconds_total": 0.0,
            "processing_wall_seconds": 0.0,
            "wall_seconds": 0.0,
            "degrade_reports": [],
            "max_length": int(max_length),
        }

    last_error: Exception | None = None
    degrade_reports: list[dict[str, Any]] = []
    for profile in _select_profiles(concurrency=concurrency, use_http2=use_http2):
        try:
            ordered_vectors, profile_meta = _run_httpx_profile(
                texts=prepared_texts,
                token_counts=ordered_token_counts,
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                profile=profile,
                chunk_size=chunk_size,
                max_retries=max_retries,
                base_backoff_seconds=base_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                progress=progress,
                progress_label=f"{progress_label} [{profile.name}]",
            )
            restored = np.zeros_like(ordered_vectors)
            restored[order] = ordered_vectors
            profile_meta["wall_seconds"] = float(perf_counter() - started_at)
            profile_meta["degrade_reports"] = degrade_reports
            profile_meta["max_length"] = int(max_length)
            profile_meta["generation_mode"] = "remote_httpx_only"
            profile_meta["resolved_device"] = f"remote:{provider}"
            profile_meta["requested_device"] = "hf"
            profile_meta["runtime_backend"] = "hf_httpx"
            return restored.astype(np.float32, copy=False), profile_meta
        except Exception as exc:
            last_error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
            degrade_reports.append(
                {
                    "profile_name": profile.name,
                    "http2_requested": bool(profile.http2_requested),
                    "concurrency": int(profile.concurrency),
                    "verify": bool(profile.verify),
                    "error": str(exc),
                }
            )
            continue

    raise RuntimeError(
        "HF HTTPX transport failed after exhausting all product profiles: "
        + "; ".join(f"{row['profile_name']}: {row['error']}" for row in degrade_reports)
    ) from last_error
