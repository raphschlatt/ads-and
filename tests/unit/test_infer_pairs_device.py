import numpy as np
import pandas as pd
import pytest

from src.approaches.nand import infer_pairs


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

    monkeypatch.setattr(infer_pairs, "_resolve_device", lambda _torch, device: "cuda" if device == "auto" else device)
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

    monkeypatch.setattr(infer_pairs, "_resolve_device", lambda _torch, device: device)
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
