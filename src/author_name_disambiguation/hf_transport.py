from __future__ import annotations

import concurrent.futures
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from author_name_disambiguation.common.cli_ui import loop_progress
from author_name_disambiguation.embedding_contract import DEFAULT_TEXT_MODEL_NAME, TEXT_EMBEDDING_DIM
from author_name_disambiguation.features.specter_runtime import compute_token_length_order, load_tokenizer_prefer_fast

DEFAULT_HF_TOKEN_ENV_VAR = "HF_TOKEN"
DEFAULT_HF_CHUNK_SIZE = 64
DEFAULT_HF_CONCURRENCY = 8
DEFAULT_HF_MAX_RETRIES = 5
DEFAULT_HF_BASE_BACKOFF_SECONDS = 1.0
DEFAULT_HF_MAX_BACKOFF_SECONDS = 30.0
DEFAULT_HF_WAIT_TIMEOUT_SECONDS = 1800
DEFAULT_HF_REQUEST_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class HFEndpointSpec:
    repository: str = DEFAULT_TEXT_MODEL_NAME
    revision: str = "main"
    framework: str = "pytorch"
    task: str = "feature-extraction"
    accelerator: str = "gpu"
    vendor: str = "aws"
    region: str = "eu-west-1"
    instance_type: str = "nvidia-t4"
    instance_size: str = "x1"
    endpoint_type: str = "protected"
    hourly_cost_usd: float = 0.5


DEFAULT_HF_ENDPOINT_SPEC = HFEndpointSpec()


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
        "Export a token with inference.endpoints.write before running HF endpoint mode."
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

    if any(code in {408, 409, 429} or 500 <= code < 600 for code in status_candidates):
        return True

    text = str(exc).lower()
    return any(token in text for token in ("429", "408", "409", "rate limit", "timeout", "503", "502", "500"))


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


def _truncate_text_to_cap(text: str, *, tokenizer: Any, cap: int) -> tuple[str, int, int]:
    truncated = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=int(cap),
        add_special_tokens=True,
    )
    token_ids = list(truncated["input_ids"])
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    if cls_token_id is not None and token_ids and int(token_ids[0]) == int(cls_token_id):
        token_ids = token_ids[1:]
    if sep_token_id is not None and token_ids and int(token_ids[-1]) == int(sep_token_id):
        token_ids = token_ids[:-1]
    truncated_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    retokenized = tokenizer(truncated_text, padding=False, truncation=False, add_special_tokens=True)
    return truncated_text, int(len(truncated["input_ids"])), int(len(retokenized["input_ids"]))


def _batched(items: Sequence[int], batch_size: int) -> Iterable[list[int]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


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


def _import_requests():
    try:
        import requests

        return requests
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "HF endpoint mode requires product dependency `requests`. "
            "Reinstall the package with current project dependencies."
        ) from exc


def _build_hf_api(*, token: str):
    try:
        from huggingface_hub import HfApi

        return HfApi(token=token)
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "HF endpoint mode requires product dependency `huggingface_hub`. "
            "Reinstall the package with current project dependencies."
        ) from exc


def _raise_hf_endpoint_error(prefix: str, exc: BaseException) -> None:
    detail = str(exc).strip()
    if "inference.endpoints.write" in detail:
        raise RuntimeError(
            f"{prefix}: missing Hugging Face permission `inference.endpoints.write`. "
            "Use a token with write access to create and delete dedicated endpoints."
        ) from exc
    raise RuntimeError(f"{prefix}: {type(exc).__name__}: {detail}") from exc


def _build_endpoint_name() -> str:
    return f"specter-t4-{uuid.uuid4().hex[:8]}"


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


def _estimate_billed_minutes(total_seconds: float) -> int:
    return max(1, int(math.ceil(max(0.0, float(total_seconds)) / 60.0)))


def _request_hf_batch_sync(
    *,
    session: Any,
    url: str,
    texts: Sequence[str],
    timeout_seconds: float,
    max_retries: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    texts_batch = [str(text) for text in texts]
    if len(texts_batch) == 0:
        return np.zeros((0, TEXT_EMBEDDING_DIM), dtype=np.float32), {
            "attempts": 0,
            "backoff_seconds_total": 0.0,
            "wall_seconds": 0.0,
            "raw_shape": [0, TEXT_EMBEDDING_DIM],
            "batch_size": 0,
        }

    requests = _import_requests()
    started_at = perf_counter()
    attempts = 0
    backoff_seconds_total = 0.0
    request_payload = {"inputs": texts_batch[0] if len(texts_batch) == 1 else texts_batch}
    while True:
        attempts += 1
        try:
            response = session.post(url, json=request_payload, timeout=float(timeout_seconds))
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
                if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
                    body = exc.response.text.strip()
                    if body:
                        detail = f"HTTP {exc.response.status_code}: {body[:240]}"
                raise RuntimeError(
                    f"HF endpoint batch failed after {attempts} attempt(s) "
                    f"for {len(texts_batch)} text(s): {type(exc).__name__}: {detail}"
                ) from exc
            delay = min(float(max_backoff_seconds), float(base_backoff_seconds) * (2 ** (attempts - 1)))
            time.sleep(delay)
            backoff_seconds_total += float(delay)


def _run_endpoint_requests(
    *,
    url: str,
    api_key: str,
    texts: Sequence[str],
    token_counts: np.ndarray,
    chunk_size: int,
    concurrency: int,
    timeout_seconds: float,
    max_retries: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    progress: bool,
    progress_label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    requests = _import_requests()
    texts_total = len(texts)
    vectors = np.full((texts_total, TEXT_EMBEDDING_DIM), np.nan, dtype=np.float32)
    raw_shapes: list[str | None] = [None] * texts_total
    per_item_wall_seconds = np.full((texts_total,), np.nan, dtype=np.float64)
    attempts_total = 0
    backoff_seconds_total = 0.0
    request_batch_size = max(1, int(chunk_size))
    worker_count = max(1, int(concurrency))
    batch_indices_list = [batch for batch in _batched(list(range(texts_total)), batch_size=request_batch_size) if batch]
    thread_local = threading.local()

    def _get_session():
        session = getattr(thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            setattr(thread_local, "session", session)
        return session

    def _run_batch(batch_indices: list[int]) -> tuple[list[int], np.ndarray, dict[str, Any]]:
        session = _get_session()
        batch_vectors, meta = _request_hf_batch_sync(
            session=session,
            url=url,
            texts=[str(texts[idx]) for idx in batch_indices],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
        return batch_indices, batch_vectors, meta

    started_at = perf_counter()
    with loop_progress(
        total=texts_total,
        label=progress_label,
        enabled=bool(progress),
        unit="text",
    ) as tracker:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_batch, batch_indices) for batch_indices in batch_indices_list]
            for future in concurrent.futures.as_completed(futures):
                batch_indices, batch_vectors, meta = future.result()
                for offset, idx in enumerate(batch_indices):
                    vectors[idx] = batch_vectors[offset]
                    per_item_wall_seconds[idx] = float(meta.get("wall_seconds") or 0.0) / max(1, len(batch_indices))
                    raw_shapes[idx] = None if meta.get("raw_shape") is None else str(meta["raw_shape"])
                attempts_total += int(meta.get("attempts") or 0)
                backoff_seconds_total += float(meta.get("backoff_seconds_total") or 0.0)
                tracker.update(len(batch_indices))

    return vectors, {
        "transport": "hf_endpoint",
        "api_concurrency": int(worker_count),
        "request_batch_size": int(request_batch_size),
        "texts_total": int(texts_total),
        "texts_successful": int(texts_total),
        "texts_failed": 0,
        "attempts_total": int(attempts_total),
        "backoff_seconds_total": float(backoff_seconds_total),
        "processing_wall_seconds": float(perf_counter() - started_at),
        "mean_sent_token_count": None if token_counts.size == 0 else float(np.mean(token_counts.astype(np.float64))),
        "max_sent_token_count": None if token_counts.size == 0 else int(np.max(token_counts)),
        "raw_shapes_top": [{"value": value, "count": int(raw_shapes.count(value))} for value in sorted({v for v in raw_shapes if v})],
        "per_item_wall_seconds_mean": None
        if not np.isfinite(per_item_wall_seconds).any()
        else float(np.nanmean(per_item_wall_seconds)),
    }


def embed_texts_via_hf_endpoint(
    *,
    texts: Sequence[str],
    model_name: str = DEFAULT_TEXT_MODEL_NAME,
    hf_token_env_var: str = DEFAULT_HF_TOKEN_ENV_VAR,
    max_length: int = 256,
    chunk_size: int = DEFAULT_HF_CHUNK_SIZE,
    concurrency: int = DEFAULT_HF_CONCURRENCY,
    max_retries: int = DEFAULT_HF_MAX_RETRIES,
    base_backoff_seconds: float = DEFAULT_HF_BASE_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_HF_MAX_BACKOFF_SECONDS,
    progress: bool = False,
    progress_label: str = "HF remote SPECTER",
    endpoint_spec: HFEndpointSpec = DEFAULT_HF_ENDPOINT_SPEC,
    wait_timeout_seconds: int = DEFAULT_HF_WAIT_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_HF_REQUEST_TIMEOUT_SECONDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = normalize_model_name(model_name)
    api_key = resolve_hf_token(hf_token_env_var)
    prepared_texts, order, ordered_token_counts = _prepare_truncated_texts(
        texts=texts,
        model_name=model_name,
        max_length=max_length,
    )
    if len(prepared_texts) == 0:
        return np.zeros((0, TEXT_EMBEDDING_DIM), dtype=np.float32), {
            "transport": "hf_endpoint",
            "runtime_backend": "hf_endpoint",
            "generation_mode": "remote_endpoint_only",
            "resolved_device": "remote:hf-endpoint",
            "requested_device": "hf",
            "texts_total": 0,
            "texts_successful": 0,
            "texts_failed": 0,
            "startup_seconds": 0.0,
            "processing_wall_seconds": 0.0,
            "delete_seconds": 0.0,
            "wall_seconds": 0.0,
            "estimated_billed_minutes": 0,
            "estimated_cost_usd": 0.0,
            "endpoint_spec": asdict(endpoint_spec),
            "max_length": int(max_length),
        }

    api = _build_hf_api(token=api_key)
    endpoint_name = _build_endpoint_name()
    endpoint_created = False
    startup_seconds = 0.0
    delete_seconds = 0.0
    endpoint_started_at = perf_counter()
    restored_vectors: np.ndarray | None = None
    result_meta: dict[str, Any] | None = None
    try:
        try:
            endpoint = api.create_inference_endpoint(
                name=endpoint_name,
                repository=endpoint_spec.repository,
                revision=endpoint_spec.revision,
                framework=endpoint_spec.framework,
                task=endpoint_spec.task,
                accelerator=endpoint_spec.accelerator,
                vendor=endpoint_spec.vendor,
                region=endpoint_spec.region,
                instance_type=endpoint_spec.instance_type,
                instance_size=endpoint_spec.instance_size,
                type=endpoint_spec.endpoint_type,
            )
            endpoint_created = True
        except Exception as exc:
            _raise_hf_endpoint_error("Creating HF endpoint failed", exc)

        try:
            endpoint = endpoint.wait(timeout=int(wait_timeout_seconds), refresh_every=5)
        except Exception as exc:
            _raise_hf_endpoint_error("Waiting for HF endpoint failed", exc)

        startup_seconds = float(perf_counter() - endpoint_started_at)
        endpoint_url = str(getattr(endpoint, "url", "") or "").strip()
        if endpoint_url == "":
            raise RuntimeError("HF endpoint did not expose a usable URL after reaching running state.")

        ordered_vectors, request_meta = _run_endpoint_requests(
            url=endpoint_url,
            api_key=api_key,
            texts=prepared_texts,
            token_counts=ordered_token_counts,
            chunk_size=chunk_size,
            concurrency=concurrency,
            timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            progress=progress,
            progress_label=progress_label,
        )
        restored = np.zeros_like(ordered_vectors)
        restored[order] = ordered_vectors
        restored_vectors = restored.astype(np.float32, copy=False)
        result_meta = {
            **request_meta,
            "runtime_backend": "hf_endpoint",
            "generation_mode": "remote_endpoint_only",
            "resolved_device": "remote:hf-endpoint",
            "requested_device": "hf",
            "endpoint_name": endpoint_name,
            "endpoint_spec": asdict(endpoint_spec),
            "startup_seconds": startup_seconds,
            "max_length": int(max_length),
        }
    finally:
        if endpoint_created:
            delete_started_at = perf_counter()
            try:
                api.delete_inference_endpoint(name=endpoint_name)
            except Exception:
                pass
            delete_seconds = float(perf_counter() - delete_started_at)
    if restored_vectors is None or result_meta is None:
        raise RuntimeError("HF endpoint transport exited without producing embeddings.")
    wall_seconds = float(startup_seconds + float(result_meta.get("processing_wall_seconds") or 0.0) + delete_seconds)
    billed_minutes = _estimate_billed_minutes(wall_seconds)
    result_meta["delete_seconds"] = delete_seconds
    result_meta["wall_seconds"] = wall_seconds
    result_meta["estimated_billed_minutes"] = billed_minutes
    result_meta["estimated_cost_usd"] = float(billed_minutes * (float(endpoint_spec.hourly_cost_usd) / 60.0))
    return restored_vectors, result_meta
