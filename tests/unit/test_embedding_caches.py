from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from author_name_disambiguation.features import embed_chars2vec, embed_specter


def _install_fake_chars2vec_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    predict_impl=None,
    direct_call_impl=None,
):
    state = {"predict_calls": [], "direct_call_calls": []}

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

    class _FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)

        def numpy(self):
            return self._array

    class _FakeEmbeddingModel:
        def predict(self, inputs, batch_size=None, verbose=None, callbacks=None):
            state["predict_calls"].append(
                {
                    "batch_size": batch_size,
                    "verbose": verbose,
                    "callback_count": 0 if callbacks is None else len(callbacks),
                    "row_count": len(inputs[0]),
                }
            )
            if predict_impl is not None:
                return predict_impl(
                    inputs,
                    batch_size=batch_size,
                    verbose=verbose,
                    callbacks=callbacks,
                    state=state,
                )
            total_rows = len(inputs[0])
            return np.ones((total_rows, 50), dtype=np.float32)

        def __call__(self, inputs, training=False):
            state["direct_call_calls"].append(
                {
                    "training": bool(training),
                    "row_count": len(inputs),
                }
            )
            if direct_call_impl is not None:
                return direct_call_impl(inputs, training=training, state=state)
            total_rows = len(inputs)
            return _FakeTensor(np.ones((total_rows, 50), dtype=np.float32))

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
    return state


def _install_fake_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    growth_raises: bool = False,
    clear_raises: bool = False,
    visible_gpus: bool = True,
):
    state = {"memory_growth_calls": [], "clear_calls": 0, "visible_gpus": ["GPU:0"] if visible_gpus else []}

    class _Experimental:
        def set_memory_growth(self, gpu, enabled):
            if growth_raises:
                raise RuntimeError("growth boom")
            state["memory_growth_calls"].append((gpu, enabled))

    class _Config:
        experimental = _Experimental()

        @staticmethod
        def list_physical_devices(kind):
            if kind != "GPU":
                return []
            return ["GPU:0"] if visible_gpus else []

        @staticmethod
        def get_visible_devices(kind):
            if kind != "GPU":
                return []
            return list(state["visible_gpus"])

        @staticmethod
        def set_visible_devices(devices, kind):
            if kind != "GPU":
                return
            state["visible_gpus"] = list(devices)

    class _Backend:
        @staticmethod
        def clear_session():
            state["clear_calls"] += 1
            if clear_raises:
                raise RuntimeError("cleanup boom")

    class _SysConfig:
        @staticmethod
        def get_build_info():
            return {
                "is_cuda_build": True,
                "cuda_version": "12.6.1",
                "cudnn_version": "9",
            }

    fake_tf = SimpleNamespace(
        __version__="2.20.0",
        config=_Config(),
        sysconfig=_SysConfig(),
        keras=SimpleNamespace(backend=_Backend()),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    return state


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
            "canonical_record_id": [0, 1],
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
            "canonical_record_id": [0, 1],
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


def test_generate_chars2vec_embeddings_defaults_to_historical_batch_32(monkeypatch: pytest.MonkeyPatch):
    _install_fake_tensorflow(monkeypatch)
    state = _install_fake_chars2vec_backend(monkeypatch)

    names = [f"name_{idx:04d}" for idx in range(70)]
    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=names,
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (70, 50)
    assert meta["execution_mode"] == "predict"
    assert meta["requested_batch_size"] == 32
    assert meta["effective_batch_size"] == 32
    assert meta["predict_batch_count"] == 3
    assert meta["oom_retry_count"] == 0
    assert len(state["predict_calls"]) == 1
    assert state["predict_calls"][0]["batch_size"] == 32


def test_generate_chars2vec_embeddings_auto_batches_on_gpu_with_callbacks(monkeypatch: pytest.MonkeyPatch):
    progress_updates: list[int] = []
    _install_fake_tensorflow(monkeypatch)

    @contextmanager
    def _fake_loop_progress(**_kwargs):
        class _Tracker:
            def update(self, n=1):
                progress_updates.append(int(n))

        yield _Tracker()

    monkeypatch.setattr(embed_chars2vec, "loop_progress", _fake_loop_progress)

    def _predict_impl(inputs, batch_size=None, verbose=None, callbacks=None, state=None):
        assert state is not None
        total_rows = len(inputs[0])
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

    state = _install_fake_chars2vec_backend(monkeypatch, predict_impl=_predict_impl)

    names = [f"name_{idx:04d}" for idx in range(900)]
    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=names,
        batch_size=None,
        show_progress=True,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (900, 50)
    assert meta["generation_mode"] == "chars2vec"
    assert meta["name_count"] == 900
    assert meta["unique_name_count"] == 900
    assert meta["wall_seconds"] >= 0.0
    assert meta["generation_seconds"] >= 0.0
    assert meta["model_load_seconds"] >= 0.0
    assert meta["normalize_seconds"] >= 0.0
    assert meta["unique_seconds"] >= 0.0
    assert meta["pad_seconds"] >= 0.0
    assert meta["predict_seconds"] >= 0.0
    assert meta["materialize_seconds"] >= 0.0
    assert meta["execution_mode"] == "predict"
    assert meta["requested_batch_size"] is None
    assert meta["effective_batch_size"] == 512
    assert meta["predict_batch_count"] == 2
    assert meta["oom_retry_count"] == 0
    assert meta["runtime_backend"] == "tensorflow-gpu"
    assert meta["tensorflow_runtime"]["status"] == "ok"
    assert len(state["predict_calls"]) == 1
    assert state["predict_calls"][0]["batch_size"] == 512
    assert state["predict_calls"][0]["verbose"] == 0
    assert state["predict_calls"][0]["callback_count"] == 1
    assert progress_updates == [1, 1]


def test_generate_chars2vec_embeddings_auto_batches_on_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(embed_chars2vec, "_configure_tensorflow_memory_growth", lambda: (None, None))
    monkeypatch.setattr(
        embed_chars2vec,
        "probe_tensorflow_runtime",
        lambda force_cpu=False: {
            "status": "cpu_fallback",
            "reason": "forced_cpu" if force_cpu else "torch_cuda_unavailable",
            "torch_cuda_available": False,
            "runtime_backend": "tensorflow-cpu",
            "tensorflow_force_cpu_requested": bool(force_cpu),
            "tensorflow_visible_gpu_count": 0,
        },
    )
    state = _install_fake_chars2vec_backend(monkeypatch)

    names = [f"name_{idx:03d}" for idx in range(140)]
    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=names,
        batch_size=None,
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (140, 50)
    assert meta["execution_mode"] == "predict"
    assert meta["requested_batch_size"] is None
    assert meta["effective_batch_size"] == 128
    assert meta["predict_batch_count"] == 2
    assert meta["oom_retry_count"] == 0
    assert meta["runtime_backend"] == "tensorflow-cpu"
    assert meta["tensorflow_runtime"]["status"] == "cpu_fallback"
    assert len(state["predict_calls"]) == 1
    assert state["predict_calls"][0]["batch_size"] == 128


def test_generate_chars2vec_embeddings_force_cpu_hides_tensorflow_gpus(monkeypatch: pytest.MonkeyPatch):
    tf_state = _install_fake_tensorflow(monkeypatch)
    _install_fake_chars2vec_backend(monkeypatch)

    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=["Doe J", "Roe A"],
        batch_size=None,
        force_cpu=True,
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (2, 50)
    assert meta["runtime_backend"] == "tensorflow-cpu"
    assert meta["force_cpu_requested"] is True
    assert meta["tensorflow_force_cpu_error"] is None
    assert meta["tensorflow_memory_growth_enabled"] is None
    assert meta["tensorflow_runtime"]["status"] == "cpu_fallback"
    assert meta["tensorflow_runtime"]["reason"] == "forced_cpu"
    assert meta["tensorflow_runtime"]["tensorflow_visible_gpu_count"] == 0
    assert tf_state["memory_growth_calls"] == []


def test_generate_chars2vec_embeddings_passes_manual_batch_size_through(monkeypatch: pytest.MonkeyPatch):
    state = _install_fake_tensorflow(monkeypatch)
    del state
    backend_state = _install_fake_chars2vec_backend(monkeypatch)

    names = [f"name_{idx:03d}" for idx in range(70)]
    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=names,
        batch_size=64,
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (70, 50)
    assert meta["requested_batch_size"] == 64
    assert meta["effective_batch_size"] == 64
    assert meta["predict_batch_count"] == 2
    assert meta["oom_retry_count"] == 0
    assert backend_state["predict_calls"][0]["batch_size"] == 64


def test_generate_chars2vec_embeddings_retries_gpu_oom_until_min_batch(monkeypatch: pytest.MonkeyPatch):
    _install_fake_tensorflow(monkeypatch)
    attempted_batches: list[int] = []

    def _predict_impl(inputs, batch_size=None, verbose=None, callbacks=None, state=None):
        del inputs, verbose, callbacks, state
        batch = int(batch_size)
        attempted_batches.append(batch)
        if batch > 32:
            raise RuntimeError(f"ResourceExhaustedError: OOM when allocating tensor for batch {batch}")
        return np.ones((3, 50), dtype=np.float32)

    _install_fake_chars2vec_backend(monkeypatch, predict_impl=_predict_impl)

    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=["Doe J", "Roe A", "Moe B"],
        batch_size=128,
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (3, 50)
    assert attempted_batches == [128, 64, 32]
    assert meta["execution_mode"] == "predict"
    assert meta["requested_batch_size"] == 128
    assert meta["effective_batch_size"] == 32
    assert meta["predict_batch_count"] == 1
    assert meta["oom_retry_count"] == 2


def test_generate_chars2vec_embeddings_direct_call_bypasses_predict(monkeypatch: pytest.MonkeyPatch):
    def _predict_impl(inputs, batch_size=None, verbose=None, callbacks=None, state=None):
        del inputs, batch_size, verbose, callbacks, state
        raise AssertionError("predict should not be used in direct_call mode")

    def _direct_call_impl(inputs, training=False, state=None):
        del state
        total_rows = len(inputs)
        out = np.zeros((total_rows, 50), dtype=np.float32)
        for idx in range(total_rows):
            out[idx, 0] = float(idx + 1)
        return SimpleNamespace(numpy=lambda: out, training=training)

    state = _install_fake_chars2vec_backend(
        monkeypatch,
        predict_impl=_predict_impl,
        direct_call_impl=_direct_call_impl,
    )

    names = ["Beta", "Alpha", "beta", "Gamma", "ALPHA"]
    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=names,
        execution_mode="direct_call",
        show_progress=True,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (5, 50)
    assert meta["execution_mode"] == "direct_call"
    assert meta["requested_batch_size"] == 32
    assert meta["effective_batch_size"] == 3
    assert meta["predict_batch_count"] == 0
    assert meta["oom_retry_count"] == 0
    assert state["predict_calls"] == []
    assert state["direct_call_calls"] == [{"training": False, "row_count": 3}]
    np.testing.assert_allclose(out[0], out[2])
    np.testing.assert_allclose(out[1], out[4])
    assert out[0, 0] != out[1, 0]
    assert out[1, 0] != out[3, 0]


def test_generate_chars2vec_embeddings_restores_input_order_from_unique_inverse(monkeypatch: pytest.MonkeyPatch):
    class _FakeEmbeddingModel:
        def predict(self, inputs, batch_size=None, verbose=None, callbacks=None):
            del batch_size, verbose, callbacks
            total_rows = len(inputs[0])
            out = np.zeros((total_rows, 50), dtype=np.float32)
            for idx in range(total_rows):
                out[idx, 0] = float(idx + 1)
            return out

    class _FakeModel:
        def __init__(self):
            self.cache: dict[str, np.ndarray] = {}
            self.char_to_ix = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789_")}
            self.vocab_size = len(self.char_to_ix)
            self.embedding_model = _FakeEmbeddingModel()

    def _pad_sequences(sequences, maxlen=None):
        del maxlen
        max_length = max(seq.shape[0] for seq in sequences)
        vocab_size = sequences[0].shape[1]
        out = np.zeros((len(sequences), max_length, vocab_size), dtype=np.float32)
        for idx, seq in enumerate(sequences):
            out[idx, : seq.shape[0], :] = seq
        return out

    monkeypatch.setitem(
        sys.modules,
        "chars2vec",
        SimpleNamespace(
            load_model=lambda _model_name: _FakeModel(),
            keras=SimpleNamespace(
                preprocessing=SimpleNamespace(sequence=SimpleNamespace(pad_sequences=_pad_sequences)),
                callbacks=SimpleNamespace(Callback=object),
            ),
        ),
    )

    names = ["Beta", "Alpha", "beta", "Gamma", "ALPHA"]
    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=names,
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (5, 50)
    assert meta["unique_name_count"] == 3
    np.testing.assert_allclose(out[0], out[2])
    np.testing.assert_allclose(out[1], out[4])
    assert out[0, 0] != out[1, 0]
    assert out[1, 0] != out[3, 0]


def test_get_or_create_chars2vec_embeddings_cache_hit_ignores_inference_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mentions = pd.DataFrame({"author_raw": ["Doe J", "Roe A"]})
    cache_path = tmp_path / "chars2vec.npy"
    cached = np.arange(100, dtype=np.float32).reshape(2, 50)
    np.save(cache_path, cached)

    def _unexpected_generate(**_kwargs):
        raise AssertionError("cache hit should skip chars2vec generation")

    monkeypatch.setattr(embed_chars2vec, "generate_chars2vec_embeddings", _unexpected_generate)

    out, meta = embed_chars2vec.get_or_create_chars2vec_embeddings(
        mentions=mentions,
        output_path=cache_path,
        batch_size=777,
        execution_mode="direct_call",
        return_meta=True,
    )

    np.testing.assert_allclose(out, cached)
    assert meta["cache_hit"] is True
    assert meta["generation_mode"] == "cache"
    assert meta["execution_mode"] == "direct_call"
    assert meta["requested_batch_size"] == 777
    assert meta["effective_batch_size"] is None
    assert meta["predict_batch_count"] == 0
    assert meta["oom_retry_count"] == 0


def test_generate_chars2vec_embeddings_enables_tensorflow_memory_growth_and_cleanup(monkeypatch: pytest.MonkeyPatch):
    tf_state = _install_fake_tensorflow(monkeypatch)
    _install_fake_chars2vec_backend(monkeypatch)

    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=["Doe J", "Roe A"],
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (2, 50)
    assert meta["tensorflow_memory_growth_enabled"] is True
    assert meta["tensorflow_memory_growth_error"] is None
    assert meta["tensorflow_cleanup_attempted"] is True
    assert meta["tensorflow_cleanup_error"] is None
    assert meta["runtime_backend"] == "tensorflow-gpu"
    assert meta["tensorflow_runtime"]["status"] == "ok"
    assert meta["wall_seconds"] >= meta["generation_seconds"] >= 0.0
    assert tf_state["memory_growth_calls"] == [("GPU:0", True)]
    assert tf_state["clear_calls"] == 1


def test_generate_chars2vec_embeddings_filters_known_tensorflow_startup_noise_for_full_chars2vec_run(
    monkeypatch: pytest.MonkeyPatch,
):
    filter_state = {"active": False}
    call_states: dict[str, bool] = {}

    @contextmanager
    def _fake_filter_known_library_stderr(*, enabled: bool):
        assert enabled is True
        filter_state["active"] = True
        try:
            yield
        finally:
            filter_state["active"] = False

    class _FakeEmbeddingModel:
        def predict(self, inputs, batch_size=None, verbose=None, callbacks=None):
            del inputs, batch_size, verbose, callbacks
            call_states["predict_filter_active"] = bool(filter_state["active"])
            return np.ones((1, 50), dtype=np.float32)

    class _FakeModel:
        def __init__(self):
            self.cache: dict[str, np.ndarray] = {}
            self.char_to_ix = {"a": 0}
            self.vocab_size = 1
            self.embedding_model = _FakeEmbeddingModel()

    def _configure_tensorflow_memory_growth():
        call_states["memory_growth_filter_active"] = bool(filter_state["active"])
        return True, None

    def _load_model(_model_name):
        call_states["load_filter_active"] = bool(filter_state["active"])
        return _FakeModel()

    def _cleanup_tensorflow_runtime(_model):
        call_states["cleanup_filter_active"] = bool(filter_state["active"])
        return None

    monkeypatch.setattr(
        embed_chars2vec,
        "probe_tensorflow_runtime",
        lambda force_cpu=False: {
            "status": "ok",
            "reason": None,
            "torch_cuda_available": True,
            "runtime_backend": "tensorflow-cpu" if force_cpu else "tensorflow-gpu",
            "tensorflow_visible_gpu_count": 0 if force_cpu else 1,
            "tensorflow_force_cpu_requested": bool(force_cpu),
        },
    )

    monkeypatch.setattr(embed_chars2vec, "_filter_known_library_stderr", _fake_filter_known_library_stderr)
    monkeypatch.setattr(embed_chars2vec, "_configure_tensorflow_memory_growth", _configure_tensorflow_memory_growth)
    monkeypatch.setattr(embed_chars2vec, "_cleanup_tensorflow_runtime", _cleanup_tensorflow_runtime)
    monkeypatch.setattr(embed_chars2vec, "_pad_sequences", lambda _model, _words: np.ones((1, 1, 1), dtype=np.float32))
    monkeypatch.setitem(
        sys.modules,
        "chars2vec",
        SimpleNamespace(
            load_model=_load_model,
            keras=SimpleNamespace(
                preprocessing=SimpleNamespace(sequence=SimpleNamespace(pad_sequences=lambda *_args, **_kwargs: np.ones((1, 1, 1), dtype=np.float32))),
                callbacks=SimpleNamespace(Callback=object),
            ),
        ),
    )

    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=["a"],
        quiet_libraries=True,
        show_progress=False,
        return_meta=True,
    )

    assert out.shape == (1, 50)
    assert meta["generation_mode"] == "chars2vec"
    assert meta["tensorflow_memory_growth_enabled"] is True
    assert meta["tensorflow_memory_growth_error"] is None
    assert call_states["memory_growth_filter_active"] is True
    assert call_states["load_filter_active"] is True
    assert call_states["predict_filter_active"] is False
    assert call_states["cleanup_filter_active"] is False


def test_generate_chars2vec_embeddings_tolerates_tensorflow_memory_growth_and_cleanup_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_tensorflow(monkeypatch, growth_raises=True, clear_raises=True)
    _install_fake_chars2vec_backend(monkeypatch)

    out, meta = embed_chars2vec.generate_chars2vec_embeddings(
        names=["Doe J"],
        show_progress=False,
        quiet_libraries=False,
        return_meta=True,
    )

    assert out.shape == (1, 50)
    assert meta["tensorflow_memory_growth_enabled"] is False
    assert "growth boom" in str(meta["tensorflow_memory_growth_error"])
    assert meta["tensorflow_cleanup_attempted"] is True
    assert "cleanup boom" in str(meta["tensorflow_cleanup_error"])


def test_chars2vec_stderr_filter_only_suppresses_known_startup_lines():
    assert embed_chars2vec._should_filter_library_stderr_line(
        "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR"
    )
    assert embed_chars2vec._should_filter_library_stderr_line(
        "I0000 00:00:1773084365.774281   21635 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0"
    )
    assert not embed_chars2vec._should_filter_library_stderr_line("RuntimeError: chars2vec embedding generation failed")
