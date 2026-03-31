from __future__ import annotations

from contextlib import contextmanager
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from author_name_disambiguation.common.cpu_runtime import detect_cpu_limit

SPECTER_RUNTIME_BACKENDS = {"transformers", "onnx_fp32"}
_DEFAULT_INTEROP_THREADS = 1


def load_tokenizer_prefer_fast(model_name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name)


def normalize_runtime_backend(runtime_backend: str | None, *, device: str) -> str:
    value = str(runtime_backend or "transformers").strip().lower() or "transformers"
    if value not in SPECTER_RUNTIME_BACKENDS:
        raise ValueError(
            f"Unsupported specter_runtime_backend={runtime_backend!r}; expected one of {sorted(SPECTER_RUNTIME_BACKENDS)!r}."
        )
    if value == "onnx_fp32" and not str(device).startswith("cpu"):
        raise ValueError("specter_runtime_backend='onnx_fp32' is only supported on CPU.")
    return value


def cpu_limit_info() -> dict[str, Any]:
    return dict(detect_cpu_limit())


def resolve_cpu_batch_size(batch_size: int | None) -> tuple[int | None, int]:
    requested = None if batch_size is None else max(1, int(batch_size))
    if requested is not None:
        return requested, requested
    cpu_limit = int(max(1, cpu_limit_info()["cpu_limit"]))
    if cpu_limit <= 4:
        return None, 8
    if cpu_limit <= 8:
        return None, 16
    return None, 32


def resolve_cpu_thread_count(requested_threads: int | None = None) -> int:
    if requested_threads is not None:
        return int(max(1, int(requested_threads)))
    return int(max(1, cpu_limit_info()["cpu_limit"]))


@contextmanager
def temporary_torch_cpu_thread_policy(
    torch_module: Any,
    *,
    intra_op_threads: int | None = None,
    interop_threads: int = _DEFAULT_INTEROP_THREADS,
) -> Iterator[dict[str, Any]]:
    requested_intra = resolve_cpu_thread_count(intra_op_threads)
    payload = {
        "cpu_limit_detected": int(max(1, cpu_limit_info()["cpu_limit"])),
        "intra_op_threads_requested": int(requested_intra),
        "interop_threads_requested": int(max(1, int(interop_threads))),
        "intra_op_threads_previous": None,
        "interop_threads_previous": None,
        "intra_op_threads_effective": None,
        "interop_threads_effective": None,
        "interop_threads_set_failed": False,
        "interop_threads_restore_failed": False,
    }

    get_num_threads = getattr(torch_module, "get_num_threads", None)
    if callable(get_num_threads):
        try:
            payload["intra_op_threads_previous"] = int(get_num_threads())
        except Exception:
            payload["intra_op_threads_previous"] = None

    get_num_interop_threads = getattr(torch_module, "get_num_interop_threads", None)
    if callable(get_num_interop_threads):
        try:
            payload["interop_threads_previous"] = int(get_num_interop_threads())
        except Exception:
            payload["interop_threads_previous"] = None

    set_num_threads = getattr(torch_module, "set_num_threads", None)
    if callable(set_num_threads):
        try:
            set_num_threads(int(requested_intra))
        except Exception:
            pass

    set_num_interop_threads = getattr(torch_module, "set_num_interop_threads", None)
    if callable(set_num_interop_threads):
        try:
            set_num_interop_threads(int(payload["interop_threads_requested"]))
        except Exception:
            payload["interop_threads_set_failed"] = True

    if callable(get_num_threads):
        try:
            payload["intra_op_threads_effective"] = int(get_num_threads())
        except Exception:
            payload["intra_op_threads_effective"] = None
    if callable(get_num_interop_threads):
        try:
            payload["interop_threads_effective"] = int(get_num_interop_threads())
        except Exception:
            payload["interop_threads_effective"] = None

    try:
        yield payload
    finally:
        if callable(set_num_threads) and payload["intra_op_threads_previous"] is not None:
            try:
                set_num_threads(int(payload["intra_op_threads_previous"]))
            except Exception:
                pass
        if callable(set_num_interop_threads) and payload["interop_threads_previous"] is not None:
            try:
                set_num_interop_threads(int(payload["interop_threads_previous"]))
            except Exception:
                payload["interop_threads_restore_failed"] = True


def compute_token_length_order(texts: list[str], *, tokenizer: Any, max_length: int) -> np.ndarray:
    if len(texts) <= 1:
        return np.arange(len(texts), dtype=np.int32)
    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=int(max_length),
            return_length=True,
        )
        raw_lengths = encoded.get("length")
        if raw_lengths is None:
            raise KeyError("length")
        lengths = np.asarray(raw_lengths, dtype=np.int32)
    except Exception:
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=int(max_length),
        )
        lengths = np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=np.int32)
    if lengths.ndim != 1 or lengths.shape[0] != len(texts):
        raise ValueError("Tokenizer returned invalid token lengths for CPU SPECTER bucketing.")
    return np.argsort(lengths, kind="stable")


def build_onnx_cache_path(
    *,
    output_path: str | Path | None,
    model_name: str,
    max_length: int,
) -> Path:
    digest = hashlib.sha256(f"{model_name}::{int(max_length)}".encode("utf-8")).hexdigest()[:16]
    if output_path is None:
        root = Path(tempfile.gettempdir()) / "author_name_disambiguation" / "onnx"
    else:
        root = Path(output_path).expanduser().resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    return root / f"specter_{digest}_fp32_op17.onnx"


def export_specter_onnx(
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    export_path: str | Path,
    sample_text: str,
    max_length: int,
) -> Path:
    path = Path(export_path).expanduser().resolve()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = tokenizer(
        [sample_text],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=int(max_length),
    )
    input_names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in sample]
    model_args = tuple(sample[name] for name in input_names)
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["last_hidden_state"] = {0: "batch", 1: "sequence"}
    with torch_module.inference_mode():
        torch_module.onnx.export(
            model,
            model_args,
            path.as_posix(),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    return path


def build_onnx_session(*, onnx_path: str | Path, num_threads: int):
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "ONNX CPU backend requires optional dependencies `onnx` and `onnxruntime`. "
            "Install them with `uv pip install --python /home/ubuntu/.venv/bin/python --editable '.[cpu_onnx]'`."
        ) from exc

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = int(max(1, int(num_threads)))
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(Path(onnx_path).expanduser().resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
