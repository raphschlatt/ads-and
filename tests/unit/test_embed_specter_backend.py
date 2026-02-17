from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from src.features import embed_specter


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
