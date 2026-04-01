from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from author_name_disambiguation.features import specter_runtime


class _LengthTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        return {"length": [3, 1, 2][: len(texts)]}


def test_compute_token_length_order_is_stable():
    order = specter_runtime.compute_token_length_order(
        ["aaa", "b", "cc"],
        tokenizer=_LengthTokenizer(),
        max_length=256,
    )
    np.testing.assert_array_equal(order, np.asarray([1, 2, 0], dtype=np.int32))


def test_normalize_runtime_backend_rejects_onnx_for_cuda():
    with pytest.raises(ValueError, match="only supported on CPU"):
        specter_runtime.normalize_runtime_backend("onnx_fp32", device="cuda")


def test_normalize_runtime_backend_accepts_internal_hf_transport():
    assert specter_runtime.normalize_runtime_backend("hf_httpx", device="cpu") == "hf_httpx"


def test_temporary_torch_cpu_thread_policy_restores_values(monkeypatch: pytest.MonkeyPatch):
    state = {"threads": 8, "interop": 4}

    fake_torch = SimpleNamespace(
        get_num_threads=lambda: state["threads"],
        set_num_threads=lambda value: state.__setitem__("threads", int(value)),
        get_num_interop_threads=lambda: state["interop"],
        set_num_interop_threads=lambda value: state.__setitem__("interop", int(value)),
    )
    monkeypatch.setattr(specter_runtime, "cpu_limit_info", lambda: {"cpu_limit": 6})

    with specter_runtime.temporary_torch_cpu_thread_policy(fake_torch, intra_op_threads=5) as meta:
        assert meta["cpu_limit_detected"] == 6
        assert meta["intra_op_threads_requested"] == 5
        assert state["threads"] == 5
        assert state["interop"] == 1

    assert state["threads"] == 8
    assert state["interop"] == 4
