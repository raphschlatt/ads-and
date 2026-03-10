from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from author_name_disambiguation.features import embed_specter


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model_name: str):
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

    def __call__(self, chunk, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        del padding, truncation, max_length, return_tensors
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
