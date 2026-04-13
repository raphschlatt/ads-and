from __future__ import annotations

import builtins
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from author_name_disambiguation.features import embed_specter


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model_name: str, **_kwargs):
        return cls()


class _FakeAutoModel:
    @classmethod
    def from_pretrained(cls, _model_name: str):
        return cls()


class _FakeAutoAdapterModel:
    def __init__(self):
        self.loaded_adapter_name = None
        self.loaded_adapter_kwargs = None

    @classmethod
    def from_pretrained(cls, _model_name: str):
        return cls()

    def load_adapter(self, adapter_name: str, **kwargs):
        self.loaded_adapter_name = adapter_name
        self.loaded_adapter_kwargs = dict(kwargs)
        return kwargs.get("load_as", adapter_name)


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModel=_FakeAutoModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_normalize_text_backend_falls_back_with_warning():
    with pytest.warns(RuntimeWarning, match="Unknown text_backend"):
        backend = embed_specter._normalize_text_backend("mystery")
    assert backend == "transformers"


def test_load_specter_components_adapter_backend_requires_adapter_name(monkeypatch: pytest.MonkeyPatch):
    embed_specter._SPECTER_MODEL_CACHE.clear()
    _install_fake_transformers(monkeypatch)
    with pytest.raises(ValueError, match="text_adapter_name is required"):
        embed_specter._load_specter_components(
            model_name="any-model",
            reuse_model=False,
            text_backend="adapters",
            text_adapter_name=None,
        )


def test_load_specter_components_adapter_backend_raises_when_adapters_missing(monkeypatch: pytest.MonkeyPatch):
    embed_specter._SPECTER_MODEL_CACHE.clear()
    _install_fake_transformers(monkeypatch)

    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "adapters":
            raise ModuleNotFoundError("No module named 'adapters'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    with pytest.raises(RuntimeError, match="requires the `adapters` package"):
        embed_specter._load_specter_components(
            model_name="any-model",
            reuse_model=False,
            text_backend="adapters",
            text_adapter_name="allenai/specter2",
            text_adapter_alias="specter2",
        )


def test_load_specter_components_adapter_backend_loads_adapter(monkeypatch: pytest.MonkeyPatch):
    embed_specter._SPECTER_MODEL_CACHE.clear()
    _install_fake_transformers(monkeypatch)
    monkeypatch.setitem(sys.modules, "adapters", SimpleNamespace(AutoAdapterModel=_FakeAutoAdapterModel))

    tokenizer, model = embed_specter._load_specter_components(
        model_name="any-model",
        reuse_model=False,
        text_backend="adapters",
        text_adapter_name="allenai/specter2",
        text_adapter_alias="specter2",
    )
    assert isinstance(tokenizer, _FakeTokenizer)
    assert isinstance(model, _FakeAutoAdapterModel)
    assert model.loaded_adapter_name == "allenai/specter2"
    assert model.loaded_adapter_kwargs == {"source": "hf", "set_active": True, "load_as": "specter2"}


class _LengthSortingTokenizer:
    def __init__(self):
        self.seen_chunks: list[list[str]] = []

    def __call__(self, chunk, padding=True, truncation=True, max_length=256, return_tensors="pt", **kwargs):
        del padding, truncation, max_length, return_tensors
        if kwargs.get("return_length"):
            return {"length": [len(text) for text in chunk]}
        self.seen_chunks.append(list(chunk))
        lengths = torch.tensor([[len(text)] for text in chunk], dtype=torch.int64)
        return {"input_ids": lengths}


class _LengthSortingModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids):
        out = torch.zeros((input_ids.shape[0], 1, 768), dtype=torch.float32, device=input_ids.device)
        out[:, 0, 0] = input_ids[:, 0].to(torch.float32)
        return SimpleNamespace(last_hidden_state=out)


def test_generate_specter_embeddings_length_batches_and_cpu_auto_precision(monkeypatch: pytest.MonkeyPatch):
    tokenizer = _LengthSortingTokenizer()
    model = _LengthSortingModel()
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    monkeypatch.setattr(
        embed_specter,
        "_load_specter_components",
        lambda **_kwargs: (tokenizer, model),
    )

    mentions = pd.DataFrame(
        {
            "title": ["a", "aaaa", "aa"],
            "abstract": ["", "", ""],
        }
    )

    out, meta = embed_specter.generate_specter_embeddings(
        mentions=mentions,
        batch_size=2,
        device="cpu",
        precision_mode="auto",
        return_meta=True,
    )

    assert out.shape == (3, 768)
    np.testing.assert_allclose(out[:, 0], np.array([1.0, 4.0, 2.0], dtype=np.float32))
    assert tokenizer.seen_chunks == [["a", "aa"], ["aaaa"]]
    assert meta["effective_precision_mode"] == "fp32"
    assert meta["requested_batch_size"] == 2
    assert meta["effective_batch_size"] == 2
    assert meta["batches_total"] == 2
    assert meta["token_count_total"] == 3
    assert meta["max_sequence_length_observed"] == 1
    assert meta["mean_sequence_length_observed"] == 1.0
    assert meta["device_to_host_flush_batch_count"] == 1
    assert meta["device_to_host_flushes"] == 2
    assert meta["tokenizers_parallelism_setting"] == "true"


def test_resolve_effective_precision_mode_uses_fp16_for_auto_on_legacy_cuda(monkeypatch: pytest.MonkeyPatch):
    class _Cuda:
        @staticmethod
        def is_bf16_supported(*, including_emulation=True):
            return bool(including_emulation)

    torch_like = SimpleNamespace(cuda=_Cuda())

    assert embed_specter._resolve_effective_precision_mode(torch_like, "auto", "cuda:0") == "amp_fp16"


def test_resolve_effective_precision_mode_keeps_bf16_for_native_support(monkeypatch: pytest.MonkeyPatch):
    class _Cuda:
        @staticmethod
        def is_bf16_supported(*, including_emulation=True):
            return True

    torch_like = SimpleNamespace(cuda=_Cuda())

    assert embed_specter._resolve_effective_precision_mode(torch_like, "auto", "cuda:0") == "amp_bf16"


def test_configure_hf_noise_preserves_tokenizers_parallelism(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    embed_specter._configure_hf_noise(True)
    assert "TOKENIZERS_PARALLELISM" not in os.environ

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    embed_specter._configure_hf_noise(True)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"


def test_resolve_specter_batch_size_uses_gpu_memory_tiers(monkeypatch: pytest.MonkeyPatch):
    class _Props:
        def __init__(self, total_memory: int):
            self.total_memory = total_memory

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props(80 * 1024**3))
    requested, effective = embed_specter._resolve_specter_batch_size(torch, None, "cuda:0")
    assert requested is None
    assert effective == 384

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props(24 * 1024**3))
    requested, effective = embed_specter._resolve_specter_batch_size(torch, None, "cuda:0")
    assert requested is None
    assert effective == 192

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props(16 * 1024**3))
    requested, effective = embed_specter._resolve_specter_batch_size(torch, None, "cuda:0")
    assert requested is None
    assert effective == 128

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props(8 * 1024**3))
    requested, effective = embed_specter._resolve_specter_batch_size(torch, None, "cuda:0")
    assert requested is None
    assert effective == 64

    monkeypatch.setattr(embed_specter, "resolve_cpu_batch_size", lambda _batch_size: (None, 16))
    requested, effective = embed_specter._resolve_specter_batch_size(torch, None, "cpu")
    assert requested is None
    assert effective == 16

    requested, effective = embed_specter._resolve_specter_batch_size(torch, 48, "cuda:0")
    assert requested == 48
    assert effective == 48


def test_resolve_device_to_host_flush_batch_count_prefers_larger_cuda_buffers(monkeypatch: pytest.MonkeyPatch):
    class _Props:
        def __init__(self, total_memory: int):
            self.total_memory = total_memory

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props(80 * 1024**3))
    assert embed_specter._resolve_device_to_host_flush_batch_count(
        torch,
        device="cuda:0",
        effective_batch_size=384,
    ) == 12
    assert embed_specter._resolve_device_to_host_flush_batch_count(
        torch,
        device="cuda:0",
        effective_batch_size=256,
    ) == 12
    assert embed_specter._resolve_device_to_host_flush_batch_count(
        torch,
        device="cuda:0",
        effective_batch_size=128,
    ) == 8
    assert embed_specter._resolve_device_to_host_flush_batch_count(
        torch,
        device="cuda:0",
        effective_batch_size=64,
    ) == 6
    assert embed_specter._resolve_device_to_host_flush_batch_count(
        torch,
        device="cpu",
        effective_batch_size=32,
    ) == 1


class _FakeBatchTensor:
    def __init__(self, values: list[list[int]]):
        self.tensor = torch.tensor(values, dtype=torch.int64)
        self.device_label = "cpu"

    @property
    def shape(self):
        return self.tensor.shape

    def to(self, device):
        self.device_label = str(device)
        return self


class _OomBackoffTokenizer:
    def __call__(self, chunk, padding=True, truncation=True, max_length=256, return_tensors="pt", **_kwargs):
        del padding, truncation, max_length, return_tensors
        return {"input_ids": _FakeBatchTensor([[len(text)] for text in chunk])}


class _OomBackoffModel:
    def __init__(self, max_cuda_batch_size: int):
        self.max_cuda_batch_size = int(max_cuda_batch_size)
        self.calls: list[tuple[str, int]] = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids):
        device_label = getattr(input_ids, "device_label", "cpu")
        batch_size = int(input_ids.shape[0])
        self.calls.append((device_label, batch_size))
        if str(device_label).startswith("cuda") and batch_size > self.max_cuda_batch_size:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        out = torch.zeros((batch_size, 1, 768), dtype=torch.float32)
        out[:, 0, 0] = input_ids.tensor[:, 0].to(torch.float32)
        return SimpleNamespace(last_hidden_state=out)


def test_generate_specter_embeddings_flushes_cuda_batches_in_order(monkeypatch: pytest.MonkeyPatch):
    tokenizer = _OomBackoffTokenizer()
    model = _OomBackoffModel(max_cuda_batch_size=64)
    monkeypatch.setattr(embed_specter, "_load_specter_components", lambda **_kwargs: (tokenizer, model))
    monkeypatch.setattr(
        embed_specter,
        "resolve_torch_device",
        lambda torch_module, device, runtime_label: (
            "cuda:0",
            {
                "requested_device": str(device),
                "resolved_device": "cuda:0",
                "fallback_reason": None,
                "torch_version": getattr(torch_module, "__version__", None),
                "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
            },
        ),
    )

    mentions = pd.DataFrame({"title": [f"x{idx}" for idx in range(9)], "abstract": [""] * 9})
    out, meta = embed_specter.generate_specter_embeddings(
        mentions=mentions,
        batch_size=2,
        device="auto",
        precision_mode="fp32",
        return_meta=True,
    )

    assert out.shape == (9, 768)
    np.testing.assert_allclose(out[:, 0], np.array([2.0] * 9, dtype=np.float32))
    assert meta["resolved_device"] == "cuda:0"
    assert meta["batches_total"] == 5
    assert meta["device_to_host_flush_batch_count"] == 4
    assert meta["device_to_host_flushes"] == 2
    assert meta["token_count_total"] == 9
    assert meta["max_sequence_length_observed"] == 1
    assert meta["mean_sequence_length_observed"] == 1.0


def test_generate_specter_embeddings_reduces_batch_size_on_cuda_oom(monkeypatch: pytest.MonkeyPatch):
    tokenizer = _OomBackoffTokenizer()
    model = _OomBackoffModel(max_cuda_batch_size=32)
    monkeypatch.setattr(embed_specter, "_load_specter_components", lambda **_kwargs: (tokenizer, model))
    monkeypatch.setattr(
        embed_specter,
        "resolve_torch_device",
        lambda torch_module, device, runtime_label: (
            "cuda:0",
            {
                "requested_device": str(device),
                "resolved_device": "cuda:0",
                "fallback_reason": None,
                "torch_version": getattr(torch_module, "__version__", None),
                "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
            },
        ),
    )

    mentions = pd.DataFrame({"title": [f"title-{idx:03d}" for idx in range(40)], "abstract": [""] * 40})
    out, meta = embed_specter.generate_specter_embeddings(
        mentions=mentions,
        batch_size=64,
        device="auto",
        precision_mode="fp32",
        return_meta=True,
    )

    assert out.shape == (40, 768)
    assert meta["requested_batch_size"] == 64
    assert meta["effective_batch_size"] == 32
    assert meta["oom_retry_count"] == 1
    assert meta["resolved_device"] == "cuda:0"
    assert model.calls[:2] == [("cuda:0", 40), ("cuda:0", 32)]
    assert model.calls[-1] == ("cuda:0", 8)


def test_generate_specter_embeddings_can_fallback_to_cpu_after_cuda_oom(monkeypatch: pytest.MonkeyPatch):
    tokenizer = _OomBackoffTokenizer()
    model = _OomBackoffModel(max_cuda_batch_size=8)
    monkeypatch.setattr(embed_specter, "_load_specter_components", lambda **_kwargs: (tokenizer, model))
    monkeypatch.setattr(
        embed_specter,
        "resolve_torch_device",
        lambda torch_module, device, runtime_label: (
            "cuda:0",
            {
                "requested_device": str(device),
                "resolved_device": "cuda:0",
                "fallback_reason": None,
                "torch_version": getattr(torch_module, "__version__", None),
                "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
            },
        ),
    )

    mentions = pd.DataFrame({"title": [f"title-{idx:03d}" for idx in range(20)], "abstract": [""] * 20})
    out, meta = embed_specter.generate_specter_embeddings(
        mentions=mentions,
        batch_size=64,
        device="auto",
        precision_mode="auto",
        return_meta=True,
    )

    assert out.shape == (20, 768)
    assert meta["resolved_device"] == "cpu"
    assert meta["fallback_reason"] == "cuda_oom_cpu_fallback"
    assert meta["effective_precision_mode"] == "fp32"
    assert meta["effective_batch_size"] == 16
    assert meta["oom_retry_count"] == 3
    assert ("cpu", 16) in model.calls


def test_generate_specter_embeddings_explicit_cuda_raises_after_oom(monkeypatch: pytest.MonkeyPatch):
    tokenizer = _OomBackoffTokenizer()
    model = _OomBackoffModel(max_cuda_batch_size=8)
    monkeypatch.setattr(embed_specter, "_load_specter_components", lambda **_kwargs: (tokenizer, model))
    monkeypatch.setattr(
        embed_specter,
        "resolve_torch_device",
        lambda torch_module, device, runtime_label: (
            "cuda:0",
            {
                "requested_device": str(device),
                "resolved_device": "cuda:0",
                "fallback_reason": None,
                "torch_version": getattr(torch_module, "__version__", None),
                "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
            },
        ),
    )

    mentions = pd.DataFrame({"title": [f"title-{idx:03d}" for idx in range(20)], "abstract": [""] * 20})
    with pytest.raises(torch.cuda.OutOfMemoryError):
        embed_specter.generate_specter_embeddings(
            mentions=mentions,
            batch_size=16,
            device="cuda",
            precision_mode="fp32",
            return_meta=True,
        )
