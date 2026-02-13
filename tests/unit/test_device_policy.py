import pytest

from src.approaches.nand import train
from src.features import embed_specter


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


class _FakeTorch:
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


def test_embed_specter_resolve_device_auto_uses_cuda_when_probe_succeeds():
    torch_like = _FakeTorch(cuda_available=True)
    assert embed_specter._resolve_device(torch_like, "auto") == "cuda"


def test_embed_specter_resolve_device_auto_falls_back_on_probe_error():
    torch_like = _FakeTorch(
        cuda_available=True,
        current_device_exc=RuntimeError("cuda init failed"),
    )
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert embed_specter._resolve_device(torch_like, "auto") == "cpu"


def test_train_resolve_device_auto_falls_back_when_cuda_unavailable():
    torch_like = _FakeTorch(cuda_available=False)
    assert train._resolve_device(torch_like, "auto") == "cpu"


def test_train_resolve_device_explicit_cuda_is_strict():
    torch_like = _FakeTorch(cuda_available=False)
    assert train._resolve_device(torch_like, "cuda") == "cuda"
