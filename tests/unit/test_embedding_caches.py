from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from author_name_disambiguation.features import embed_chars2vec, embed_specter


def test_get_or_create_chars2vec_embeddings_rebuilds_invalid_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mentions = pd.DataFrame({"author_raw": ["Doe J", "Roe A"]})
    cache_path = tmp_path / "chars2vec.npy"
    np.save(cache_path, np.zeros((1, 1), dtype=np.float32))

    monkeypatch.setattr(
        embed_chars2vec,
        "generate_chars2vec_embeddings",
        lambda names, **_kwargs: np.ones((len(names), 50), dtype=np.float32),
    )

    with pytest.warns(RuntimeWarning, match="recomputing"):
        out = embed_chars2vec.get_or_create_chars2vec_embeddings(
            mentions=mentions,
            output_path=cache_path,
        )

    assert out.shape == (2, 50)
    assert np.load(cache_path).shape == (2, 50)


def test_get_or_create_specter_embeddings_rebuilds_invalid_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mentions = pd.DataFrame(
        {
            "title": ["Paper 1", "Paper 2"],
            "abstract": ["Abstract 1", "Abstract 2"],
        }
    )
    cache_path = tmp_path / "specter.npy"
    np.save(cache_path, np.zeros((1, 1), dtype=np.float32))

    def _generate(*, mentions, return_meta=False, **_kwargs):
        arr = np.ones((len(mentions), 768), dtype=np.float32)
        meta = {
            "cache_hit": False,
            "generation_mode": "model_only",
            "requested_device": "auto",
            "resolved_device": "cpu",
            "fallback_reason": "torch_cuda_unavailable",
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "cuda_probe_error": None,
            "model_to_cuda_error": None,
            "effective_precision_mode": None,
            "column_present": False,
            "precomputed_embedding_count": 0,
            "recomputed_embedding_count": len(mentions),
            "used_precomputed_embeddings": False,
        }
        return (arr, meta) if return_meta else arr

    monkeypatch.setattr(embed_specter, "generate_specter_embeddings", _generate)

    with pytest.warns(RuntimeWarning, match="recomputing"):
        out, meta = embed_specter.get_or_create_specter_embeddings(
            mentions=mentions,
            output_path=cache_path,
            return_meta=True,
        )

    assert out.shape == (2, 768)
    assert meta["cache_hit"] is False
    assert np.load(cache_path).shape == (2, 768)


def test_generate_specter_embeddings_uses_precomputed_vectors_directly():
    vec_a = np.linspace(0.0, 1.0, num=768, dtype=np.float32)
    vec_b = np.linspace(1.0, 2.0, num=768, dtype=np.float32)
    mentions = pd.DataFrame(
        {
            "title": ["Paper 1", "Paper 2"],
            "abstract": ["Abstract 1", "Abstract 2"],
            "precomputed_embedding": [vec_a.tolist(), vec_b.tolist()],
        }
    )

    out, meta = embed_specter.generate_specter_embeddings(
        mentions=mentions,
        prefer_precomputed=True,
        return_meta=True,
    )

    assert out.shape == (2, 768)
    np.testing.assert_allclose(out[0], vec_a)
    np.testing.assert_allclose(out[1], vec_b)
    assert meta["generation_mode"] == "precomputed_only"
    assert meta["precomputed_embedding_count"] == 2
    assert meta["recomputed_embedding_count"] == 0
    assert meta["used_precomputed_embeddings"] is True


def test_generate_chars2vec_embeddings_uses_single_predict_with_callbacks(monkeypatch: pytest.MonkeyPatch):
    predict_calls: list[dict[str, object]] = []
    progress_updates: list[int] = []

    @contextmanager
    def _fake_loop_progress(**_kwargs):
        class _Tracker:
            def update(self, n=1):
                progress_updates.append(int(n))

        yield _Tracker()

    monkeypatch.setattr(embed_chars2vec, "loop_progress", _fake_loop_progress)

    def _pad_sequences(sequences, maxlen=None):
        del maxlen
        max_length = max(seq.shape[0] for seq in sequences)
        vocab_size = sequences[0].shape[1]
        out = np.zeros((len(sequences), max_length, vocab_size), dtype=np.float32)
        for idx, seq in enumerate(sequences):
            out[idx, : seq.shape[0], :] = seq
        return out

    class _FakeCallback:
        pass

    class _FakeEmbeddingModel:
        def predict(self, inputs, batch_size=None, verbose=None, callbacks=None):
            total_rows = len(inputs[0])
            predict_calls.append(
                {
                    "batch_size": batch_size,
                    "verbose": verbose,
                    "callback_count": 0 if callbacks is None else len(callbacks),
                }
            )
            total_batches = (total_rows + int(batch_size) - 1) // int(batch_size)
            for callback in callbacks or []:
                if hasattr(callback, "on_predict_begin"):
                    callback.on_predict_begin({})
            for batch_idx in range(total_batches):
                for callback in callbacks or []:
                    if hasattr(callback, "on_predict_batch_end"):
                        callback.on_predict_batch_end(batch_idx, {})
            for callback in callbacks or []:
                if hasattr(callback, "on_predict_end"):
                    callback.on_predict_end({})
            return np.ones((total_rows, 50), dtype=np.float32)

    class _FakeModel:
        def __init__(self):
            self.cache: dict[str, np.ndarray] = {}
            self.char_to_ix = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789_")}
            self.vocab_size = len(self.char_to_ix)
            self.embedding_model = _FakeEmbeddingModel()

    fake_chars2vec = SimpleNamespace(
        load_model=lambda _model_name: _FakeModel(),
        keras=SimpleNamespace(
            preprocessing=SimpleNamespace(sequence=SimpleNamespace(pad_sequences=_pad_sequences)),
            callbacks=SimpleNamespace(Callback=_FakeCallback),
        ),
    )
    monkeypatch.setitem(sys.modules, "chars2vec", fake_chars2vec)

    names = [f"name_{idx:03d}" for idx in range(70)]
    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=names,
        show_progress=True,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (70, 50)
    assert meta["generation_mode"] == "chars2vec"
    assert len(predict_calls) == 1
    assert predict_calls[0] == {"batch_size": 32, "verbose": 0, "callback_count": 1}
    assert progress_updates == [1, 1, 1]


def test_chars2vec_stderr_filter_only_suppresses_known_startup_lines():
    assert embed_chars2vec._should_filter_library_stderr_line(
        "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR"
    )
    assert embed_chars2vec._should_filter_library_stderr_line(
        "I0000 00:00:1773084365.774281   21635 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0"
    )
    assert not embed_chars2vec._should_filter_library_stderr_line("RuntimeError: chars2vec embedding generation failed")
