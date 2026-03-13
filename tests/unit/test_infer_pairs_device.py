import numpy as np
import pandas as pd
import pytest

from author_name_disambiguation.approaches.nand import infer_pairs


class _FakeCuda:
    def __init__(self, *, available: bool, current_device_exc: Exception | None = None):
        self._available = available
        self._current_device_exc = current_device_exc

    def is_available(self) -> bool:
        return self._available

    def current_device(self) -> int:
        if self._current_device_exc is not None:
            raise self._current_device_exc
        return 0


class _FakeTorchResolve:
    def __init__(
        self,
        *,
        cuda_available: bool,
        current_device_exc: Exception | None = None,
        empty_exc: Exception | None = None,
    ):
        self.cuda = _FakeCuda(available=cuda_available, current_device_exc=current_device_exc)
        self._empty_exc = empty_exc

    def empty(self, *_args, **_kwargs):
        if self._empty_exc is not None:
            raise self._empty_exc
        return object()


def test_resolve_device_auto_uses_cuda_when_probe_succeeds():
    torch_like = _FakeTorchResolve(cuda_available=True)
    assert infer_pairs._resolve_device(torch_like, "auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_on_cuda_init_error():
    torch_like = _FakeTorchResolve(
        cuda_available=True,
        current_device_exc=RuntimeError("cuda init failed"),
    )
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert infer_pairs._resolve_device(torch_like, "auto") == "cpu"


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_unavailable():
    torch_like = _FakeTorchResolve(cuda_available=False)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert infer_pairs._resolve_device(torch_like, "auto") == "cpu"


def test_load_checkpoint_always_deserializes_on_cpu(monkeypatch):
    calls = {}

    class _FakeTorchLoad:
        def load(self, path, map_location=None):
            calls["path"] = path
            calls["map_location"] = map_location
            return {"ok": True}

    monkeypatch.setattr(infer_pairs, "_require_torch", lambda: _FakeTorchLoad())
    out = infer_pairs.load_checkpoint("checkpoint.pt", device="cuda")

    assert out == {"ok": True}
    assert calls["map_location"] == "cpu"


class _FakeModel:
    def __init__(self):
        self.to_calls = []

    def load_state_dict(self, _state_dict):
        return None

    def to(self, device):
        self.to_calls.append(str(device))
        if str(device).startswith("cuda"):
            raise RuntimeError("cuda boom")
        return self

    def eval(self):
        return self

    def __call__(self, x):
        return x


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["pair_id", "mention_id_1", "mention_id_2", "block_key"],
    )


def test_score_pairs_auto_falls_back_to_cpu_when_model_to_cuda_fails(monkeypatch):
    model = _FakeModel()

    monkeypatch.setattr(
        infer_pairs,
        "resolve_torch_device",
        lambda _torch, device, runtime_label: (
            "cuda" if device == "auto" else device,
            {
                "requested_device": str(device),
                "resolved_device": "cuda" if device == "auto" else str(device),
                "fallback_reason": None,
                "torch_version": "fake",
                "torch_cuda_version": "12.1",
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": None,
            },
        ),
    )
    monkeypatch.setattr(infer_pairs, "load_checkpoint", lambda **_kwargs: {"model_config": {}, "state_dict": {}})
    monkeypatch.setattr(infer_pairs, "create_encoder", lambda _config: model)
    monkeypatch.setattr(
        infer_pairs,
        "build_feature_matrix",
        lambda chars2vec, text_emb: np.zeros((chars2vec.shape[0], 2), dtype=np.float32),
    )

    mentions = pd.DataFrame({"mention_id": ["m1"]})
    pairs = _empty_pairs()

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        out = infer_pairs.score_pairs_with_checkpoint(
            mentions=mentions,
            pairs=pairs,
            chars2vec=np.zeros((1, 1), dtype=np.float32),
            text_emb=np.zeros((1, 1), dtype=np.float32),
            checkpoint_path="checkpoint.pt",
            device="auto",
        )

    assert model.to_calls == ["cuda", "cpu"]
    assert list(out.columns) == ["pair_id", "mention_id_1", "mention_id_2", "block_key", "cosine_sim", "distance"]
    assert len(out) == 0


def test_score_pairs_explicit_cuda_raises_when_model_to_cuda_fails(monkeypatch):
    model = _FakeModel()

    monkeypatch.setattr(
        infer_pairs,
        "resolve_torch_device",
        lambda _torch, device, runtime_label: (
            str(device),
            {
                "requested_device": str(device),
                "resolved_device": str(device),
                "fallback_reason": None,
                "torch_version": "fake",
                "torch_cuda_version": "12.1",
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": None,
            },
        ),
    )
    monkeypatch.setattr(infer_pairs, "load_checkpoint", lambda **_kwargs: {"model_config": {}, "state_dict": {}})
    monkeypatch.setattr(infer_pairs, "create_encoder", lambda _config: model)
    monkeypatch.setattr(
        infer_pairs,
        "build_feature_matrix",
        lambda chars2vec, text_emb: np.zeros((chars2vec.shape[0], 2), dtype=np.float32),
    )

    mentions = pd.DataFrame({"mention_id": ["m1"]})
    pairs = _empty_pairs()

    with pytest.raises(RuntimeError, match="cuda boom"):
        infer_pairs.score_pairs_with_checkpoint(
            mentions=mentions,
            pairs=pairs,
            chars2vec=np.zeros((1, 1), dtype=np.float32),
            text_emb=np.zeros((1, 1), dtype=np.float32),
            checkpoint_path="checkpoint.pt",
            device="cuda",
        )

    assert model.to_calls == ["cuda"]


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorchScore:
    def __init__(self):
        self.cuda = _FakeCuda(available=False)

        class _Functional:
            @staticmethod
            def cosine_similarity(z1, z2, dim=1):
                _ = z2, dim
                n = z1.arr.shape[0]
                return _FakeTensor(np.full((n,), 1.0000001, dtype=np.float32))

        class _NN:
            functional = _Functional()

        self.nn = _NN()

    def from_numpy(self, arr):
        return _FakeTensor(arr)

    def no_grad(self):
        return _NoGrad()


class _FakeTorchEncode:
    def __init__(self, *, cuda_available: bool = True):
        self.cuda = _FakeCuda(available=cuda_available)

    def from_numpy(self, arr):
        return _FakeTensor(arr)

    def no_grad(self):
        return _NoGrad()

    def cat(self, *_args, **_kwargs):
        raise AssertionError("torch.cat should not be called in _encode_mentions")


class _IdentityModel:
    def load_state_dict(self, _state_dict):
        return None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        return x


class _NormalizeModel:
    def load_state_dict(self, _state_dict):
        return None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


class _RecordToModel(_IdentityModel):
    def __init__(self):
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(str(device))
        return self


def test_score_pairs_clamps_numeric_boundary_values(monkeypatch):
    monkeypatch.setattr(infer_pairs, "_require_torch", lambda: _FakeTorchScore())
    monkeypatch.setattr(
        infer_pairs,
        "resolve_torch_device",
        lambda _torch, _device, runtime_label: (
            "cpu",
            {
                "requested_device": str(_device),
                "resolved_device": "cpu",
                "fallback_reason": None,
                "torch_version": "fake",
                "torch_cuda_version": None,
                "torch_cuda_available": False,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": None,
            },
        ),
    )
    monkeypatch.setattr(infer_pairs, "load_checkpoint", lambda **_kwargs: {"model_config": {}, "state_dict": {}})
    monkeypatch.setattr(infer_pairs, "create_encoder", lambda _config: _IdentityModel())
    monkeypatch.setattr(
        infer_pairs,
        "build_feature_matrix",
        lambda chars2vec, text_emb: np.ones((chars2vec.shape[0], 2), dtype=np.float32),
    )

    mentions = pd.DataFrame({"mention_id": ["m1", "m2"]})
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "m1__m2",
                "mention_id_1": "m1",
                "mention_id_2": "m2",
                "block_key": "a.block",
            }
        ]
    )

    out, runtime_meta = infer_pairs.score_pairs_with_checkpoint(
        mentions=mentions,
        pairs=pairs,
        chars2vec=np.zeros((2, 1), dtype=np.float32),
        text_emb=np.zeros((2, 1), dtype=np.float32),
        checkpoint_path="checkpoint.pt",
        device="cpu",
        return_runtime_meta=True,
    )

    assert float(out["cosine_sim"].iloc[0]) == pytest.approx(1.0)
    assert float(out["distance"].iloc[0]) == pytest.approx(0.0)
    assert runtime_meta["numeric_clamping"]["clamped"] is True
    assert runtime_meta["numeric_clamping"]["events"] == 1
    assert runtime_meta["numeric_clamping"]["cosine_above_max_count"] == 1


def test_encode_mentions_cuda_path_materializes_batches_on_cpu_without_torch_cat():
    torch_like = _FakeTorchEncode(cuda_available=True)
    features = np.arange(12, dtype=np.float32).reshape(3, 4)

    emb, norms = infer_pairs._encode_mentions(
        torch=torch_like,
        model=_IdentityModel(),
        features=features,
        batch_size=2,
        device="cuda",
        precision_mode="fp32",
        show_progress=False,
    )

    np.testing.assert_allclose(emb, features)
    np.testing.assert_allclose(norms, np.linalg.norm(features, axis=1).astype(np.float32))


def test_score_pairs_matches_reference_cosine_from_preencoded_mentions(monkeypatch):
    monkeypatch.setattr(infer_pairs, "load_checkpoint", lambda **_kwargs: {"model_config": {}, "state_dict": {}})
    monkeypatch.setattr(infer_pairs, "create_encoder", lambda _config: _NormalizeModel())

    mentions = pd.DataFrame({"mention_id": ["m1", "m2", "m3"]})
    pairs = pd.DataFrame(
        [
            {"pair_id": "m1__m2", "mention_id_1": "m1", "mention_id_2": "m2", "block_key": "blk.a"},
            {"pair_id": "m2__m3", "mention_id_1": "m2", "mention_id_2": "m3", "block_key": "blk.a"},
            {"pair_id": "m1__missing", "mention_id_1": "m1", "mention_id_2": "missing", "block_key": "blk.a"},
        ]
    )
    chars = np.array([[1.0, 0.0], [0.2, 0.8], [0.5, 0.5]], dtype=np.float32)
    text = np.array([[0.1, 0.9], [0.3, 0.7], [0.9, 0.1]], dtype=np.float32)

    out, runtime_meta = infer_pairs.score_pairs_with_checkpoint(
        mentions=mentions,
        pairs=pairs,
        chars2vec=chars,
        text_emb=text,
        checkpoint_path="checkpoint.pt",
        device="cpu",
        return_runtime_meta=True,
        show_progress=False,
    )

    features = infer_pairs.build_feature_matrix(chars, text)
    mention_emb = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    mention_index = {str(m): i for i, m in enumerate(mentions["mention_id"].tolist())}
    expected_pairs = pairs[pairs["mention_id_2"].isin(mention_index)].reset_index(drop=True)
    expected_sim = []
    for row in expected_pairs.itertuples(index=False):
        z1 = mention_emb[mention_index[str(row.mention_id_1)]]
        z2 = mention_emb[mention_index[str(row.mention_id_2)]]
        expected_sim.append(float(np.dot(z1, z2)))

    assert list(out["pair_id"]) == ["m1__m2", "m2__m3"]
    np.testing.assert_allclose(out["cosine_sim"].to_numpy(), np.asarray(expected_sim, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["distance"].to_numpy(), 1.0 - np.asarray(expected_sim, dtype=np.float32), rtol=1e-6, atol=1e-6)
    assert runtime_meta["pair_scoring_strategy"] == "preencoded_mentions"
    assert runtime_meta["mention_storage_device"] == "cpu"
    assert runtime_meta["cuda_oom_fallback_used"] is False
    assert runtime_meta["feature_matrix_shape"] == [3, 4]
    assert runtime_meta["mention_embedding_shape"] == [3, 4]
    assert runtime_meta["pairs_total_rows"] == 3
    assert runtime_meta["pairs_valid_rows"] == 2


def test_score_pairs_auto_retries_on_cpu_after_cuda_oom_during_mention_encoding(monkeypatch):
    model = _RecordToModel()

    monkeypatch.setattr(
        infer_pairs,
        "resolve_torch_device",
        lambda _torch, device, runtime_label: (
            "cuda" if device == "auto" else str(device),
            {
                "requested_device": str(device),
                "resolved_device": "cuda" if device == "auto" else str(device),
                "fallback_reason": None,
                "torch_version": "fake",
                "torch_cuda_version": "12.1",
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": None,
            },
        ),
    )
    monkeypatch.setattr(infer_pairs, "load_checkpoint", lambda **_kwargs: {"model_config": {}, "state_dict": {}})
    monkeypatch.setattr(infer_pairs, "create_encoder", lambda _config: model)
    monkeypatch.setattr(
        infer_pairs,
        "build_feature_matrix",
        lambda chars2vec, text_emb: np.zeros((chars2vec.shape[0], 2), dtype=np.float32),
    )

    def _fake_encode_mentions(*, device, **_kwargs):
        if str(device).startswith("cuda"):
            raise RuntimeError("CUDA out of memory while encoding mentions")
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        norms = np.array([1.0, 1.0], dtype=np.float32)
        return emb, norms

    monkeypatch.setattr(infer_pairs, "_encode_mentions", _fake_encode_mentions)

    mentions = pd.DataFrame({"mention_id": ["m1", "m2"]})
    pairs = pd.DataFrame(
        [{"pair_id": "m1__m2", "mention_id_1": "m1", "mention_id_2": "m2", "block_key": "blk.a"}]
    )

    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        out, runtime_meta = infer_pairs.score_pairs_with_checkpoint(
            mentions=mentions,
            pairs=pairs,
            chars2vec=np.zeros((2, 1), dtype=np.float32),
            text_emb=np.zeros((2, 1), dtype=np.float32),
            checkpoint_path="checkpoint.pt",
            device="auto",
            return_runtime_meta=True,
        )

    assert list(out["pair_id"]) == ["m1__m2"]
    assert model.to_calls == ["cuda", "cpu"]
    assert runtime_meta["resolved_device"] == "cpu"
    assert runtime_meta["fallback_reason"] == "pair_scoring_cuda_oom_retry_cpu"
    assert runtime_meta["cuda_oom_fallback_used"] is True


def test_score_pairs_explicit_cuda_oom_during_mention_encoding_still_raises(monkeypatch):
    model = _RecordToModel()

    monkeypatch.setattr(
        infer_pairs,
        "resolve_torch_device",
        lambda _torch, device, runtime_label: (
            str(device),
            {
                "requested_device": str(device),
                "resolved_device": str(device),
                "fallback_reason": None,
                "torch_version": "fake",
                "torch_cuda_version": "12.1",
                "torch_cuda_available": True,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": None,
            },
        ),
    )
    monkeypatch.setattr(infer_pairs, "load_checkpoint", lambda **_kwargs: {"model_config": {}, "state_dict": {}})
    monkeypatch.setattr(infer_pairs, "create_encoder", lambda _config: model)
    monkeypatch.setattr(
        infer_pairs,
        "build_feature_matrix",
        lambda chars2vec, text_emb: np.zeros((chars2vec.shape[0], 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        infer_pairs,
        "_encode_mentions",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory during mention encoding")),
    )

    mentions = pd.DataFrame({"mention_id": ["m1", "m2"]})
    pairs = pd.DataFrame(
        [{"pair_id": "m1__m2", "mention_id_1": "m1", "mention_id_2": "m2", "block_key": "blk.a"}]
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        infer_pairs.score_pairs_with_checkpoint(
            mentions=mentions,
            pairs=pairs,
            chars2vec=np.zeros((2, 1), dtype=np.float32),
            text_emb=np.zeros((2, 1), dtype=np.float32),
            checkpoint_path="checkpoint.pt",
            device="cuda",
            return_runtime_meta=True,
        )

    assert model.to_calls == ["cuda"]


def test_score_pairs_chunked_parity_matches_dataframe_path(monkeypatch, tmp_path):
    monkeypatch.setattr(infer_pairs, "load_checkpoint", lambda **_kwargs: {"model_config": {}, "state_dict": {}})
    monkeypatch.setattr(infer_pairs, "create_encoder", lambda _config: _NormalizeModel())

    mentions = pd.DataFrame({"mention_id": ["m1", "m2", "m3", "m4"]})
    pairs = pd.DataFrame(
        [
            {"pair_id": "m1__m2", "mention_id_1": "m1", "mention_id_2": "m2", "block_key": "blk.a"},
            {"pair_id": "m2__m3", "mention_id_1": "m2", "mention_id_2": "m3", "block_key": "blk.a"},
            {"pair_id": "m3__m4", "mention_id_1": "m3", "mention_id_2": "m4", "block_key": "blk.b"},
        ]
    )
    chars = np.array([[1.0, 0.0], [0.2, 0.8], [0.5, 0.5], [0.8, 0.2]], dtype=np.float32)
    text = np.array([[0.1, 0.9], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]], dtype=np.float32)

    direct, direct_meta = infer_pairs.score_pairs_with_checkpoint(
        mentions=mentions,
        pairs=pairs,
        chars2vec=chars,
        text_emb=text,
        checkpoint_path="checkpoint.pt",
        device="cpu",
        return_runtime_meta=True,
        show_progress=False,
    )

    pairs_path = tmp_path / "pairs.parquet"
    output_path = tmp_path / "scores.parquet"
    pairs.to_parquet(pairs_path, index=False)
    chunked, chunked_meta = infer_pairs.score_pairs_with_checkpoint(
        mentions=mentions,
        pairs=pairs_path,
        chars2vec=chars,
        text_emb=text,
        checkpoint_path="checkpoint.pt",
        output_path=output_path,
        batch_size=2,
        chunk_rows=2,
        device="cpu",
        return_runtime_meta=True,
        show_progress=False,
    )

    pd.testing.assert_frame_equal(direct.reset_index(drop=True), chunked.reset_index(drop=True))
    pd.testing.assert_frame_equal(chunked.reset_index(drop=True), pd.read_parquet(output_path).reset_index(drop=True))
    assert direct_meta["pair_input_mode"] == "dataframe"
    assert chunked_meta["pair_input_mode"] == "parquet_chunked"
    assert chunked_meta["parquet_read_seconds"] >= 0.0
    assert chunked_meta["pandas_conversion_seconds"] == 0.0
    assert chunked_meta["arrow_column_extract_seconds"] >= 0.0
    assert chunked_meta["pair_score_seconds"] >= 0.0
    assert chunked_meta["parquet_output_table_seconds"] >= 0.0
    assert chunked_meta["parquet_write_seconds"] >= 0.0
