from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from author_name_disambiguation import hf_transport


def test_normalize_hf_batch_response_accepts_ragged_token_level_features():
    payload = [
        [[1.0] * 768, [9.0] * 768, [7.0] * 768],
        [[2.0] * 768],
    ]

    out = hf_transport.normalize_hf_batch_response(payload, expected_items=2)

    assert out.shape == (2, 768)
    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0], dtype=np.float32))


def test_request_hf_batch_sync_uses_array_inputs_for_multi_text_batch(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [
                [[1.0] * 768, [9.0] * 768],
                [[2.0] * 768],
            ]

    class _FakeSession:
        def post(self, url, json, timeout):
            seen["url"] = url
            seen["json"] = json
            seen["timeout"] = timeout
            return _FakeResponse()

    monkeypatch.setattr(hf_transport, "_import_requests", lambda: SimpleNamespace(HTTPError=RuntimeError))

    vectors, meta = hf_transport._request_hf_batch_sync(
        session=_FakeSession(),
        url="https://endpoint.example/embed",
        texts=["alpha", "beta"],
        timeout_seconds=30.0,
        max_retries=0,
        base_backoff_seconds=0.1,
        max_backoff_seconds=1.0,
    )

    assert seen["json"] == {"inputs": ["alpha", "beta"]}
    assert vectors.shape == (2, 768)
    np.testing.assert_allclose(vectors[:, 0], np.array([1.0, 2.0], dtype=np.float32))
    assert meta["batch_size"] == 2


def test_truncate_text_to_cap_preserves_inner_sep_token():
    class _FakeTokenizer:
        cls_token_id = 101
        sep_token_id = 102

        def __call__(self, text, *, padding, truncation, max_length=None, add_special_tokens):
            if text == "orig":
                return {"input_ids": [101, 11, 102, 12, 102]}
            if text == "kept [SEP] tail":
                return {"input_ids": [101, 11, 102, 12, 102]}
            raise AssertionError(f"unexpected text {text!r}")

        def decode(self, token_ids, skip_special_tokens):
            assert skip_special_tokens is False
            assert token_ids == [11, 102, 12]
            return "kept [SEP] tail"

    truncated_text, truncated_tokens, retokenized_tokens = hf_transport._truncate_text_to_cap(
        "orig",
        tokenizer=_FakeTokenizer(),
        cap=256,
    )

    assert truncated_text == "kept [SEP] tail"
    assert truncated_tokens == 5
    assert retokenized_tokens == 5


def test_embed_texts_via_hf_endpoint_lifecycle(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(
        hf_transport,
        "_prepare_truncated_texts",
        lambda **kwargs: (["alpha", "beta"], np.asarray([0, 1], dtype=np.int64), np.asarray([3, 4], dtype=np.int64)),
    )

    class _FakeEndpoint:
        url = "https://endpoint.example"

        def wait(self, timeout=None, refresh_every=5):
            calls.append(("wait", timeout))
            return self

    class _FakeApi:
        def create_inference_endpoint(self, **kwargs):
            calls.append(("create", kwargs["name"]))
            return _FakeEndpoint()

        def delete_inference_endpoint(self, *, name):
            calls.append(("delete", name))

    class _FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [
                [[1.0] * 768, [9.0] * 768],
                [[2.0] * 768],
            ]

    class _FakeSession:
        def __init__(self):
            self.headers = {}

        def post(self, url, json, timeout):
            calls.append(("post", json))
            return _FakeResponse()

    monkeypatch.setattr(hf_transport, "_build_hf_api", lambda *, token: _FakeApi())
    monkeypatch.setattr(
        hf_transport,
        "_import_requests",
        lambda: SimpleNamespace(Session=_FakeSession, HTTPError=RuntimeError),
    )

    vectors, meta = hf_transport.embed_texts_via_hf_endpoint(texts=["alpha", "beta"], progress=False)

    assert vectors.shape == (2, 768)
    np.testing.assert_allclose(vectors[:, 0], np.array([1.0, 2.0], dtype=np.float32))
    assert meta["runtime_backend"] == "hf_endpoint"
    assert meta["generation_mode"] == "remote_endpoint_only"
    assert meta["resolved_device"] == "remote:hf-endpoint"
    assert meta["transport"] == "hf_endpoint"
    assert calls[0][0] == "create"
    assert calls[1][0] == "wait"
    assert calls[-1][0] == "delete"


def test_embed_texts_via_hf_endpoint_deletes_endpoint_after_wait_failure(monkeypatch: pytest.MonkeyPatch):
    deleted: list[str] = []

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(
        hf_transport,
        "_prepare_truncated_texts",
        lambda **kwargs: (["alpha"], np.asarray([0], dtype=np.int64), np.asarray([3], dtype=np.int64)),
    )

    class _FakeEndpoint:
        url = "https://endpoint.example"

        def wait(self, timeout=None, refresh_every=5):
            raise RuntimeError("endpoint stuck")

    class _FakeApi:
        def create_inference_endpoint(self, **kwargs):
            return _FakeEndpoint()

        def delete_inference_endpoint(self, *, name):
            deleted.append(name)

    monkeypatch.setattr(hf_transport, "_build_hf_api", lambda *, token: _FakeApi())

    with pytest.raises(RuntimeError, match="Waiting for HF endpoint failed"):
        hf_transport.embed_texts_via_hf_endpoint(texts=["alpha"], progress=False)

    assert len(deleted) == 1


def test_embed_texts_via_hf_endpoint_deletes_endpoint_after_infer_failure(monkeypatch: pytest.MonkeyPatch):
    deleted: list[str] = []

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(
        hf_transport,
        "_prepare_truncated_texts",
        lambda **kwargs: (["alpha"], np.asarray([0], dtype=np.int64), np.asarray([3], dtype=np.int64)),
    )

    class _FakeEndpoint:
        url = "https://endpoint.example"

        def wait(self, timeout=None, refresh_every=5):
            return self

    class _FakeApi:
        def create_inference_endpoint(self, **kwargs):
            return _FakeEndpoint()

        def delete_inference_endpoint(self, *, name):
            deleted.append(name)

    class _FailingSession:
        def __init__(self):
            self.headers = {}

        def post(self, url, json, timeout):
            raise RuntimeError("network broke")

    monkeypatch.setattr(hf_transport, "_build_hf_api", lambda *, token: _FakeApi())
    monkeypatch.setattr(
        hf_transport,
        "_import_requests",
        lambda: SimpleNamespace(Session=_FailingSession, HTTPError=RuntimeError),
    )

    with pytest.raises(RuntimeError, match="HF endpoint batch failed"):
        hf_transport.embed_texts_via_hf_endpoint(texts=["alpha"], progress=False, max_retries=0)

    assert len(deleted) == 1


def test_embed_texts_via_hf_endpoint_write_permission_error_is_actionable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(
        hf_transport,
        "_prepare_truncated_texts",
        lambda **kwargs: (["alpha"], np.asarray([0], dtype=np.int64), np.asarray([3], dtype=np.int64)),
    )

    class _FakeApi:
        def create_inference_endpoint(self, **kwargs):
            raise RuntimeError("403 Forbidden: missing permissions: inference.endpoints.write")

    monkeypatch.setattr(hf_transport, "_build_hf_api", lambda *, token: _FakeApi())

    with pytest.raises(RuntimeError, match="inference.endpoints.write"):
        hf_transport.embed_texts_via_hf_endpoint(texts=["alpha"], progress=False)
