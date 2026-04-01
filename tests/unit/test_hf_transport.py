from __future__ import annotations

import asyncio

import numpy as np

from author_name_disambiguation.hf_transport import (
    _request_hf_batch_async,
    normalize_hf_batch_response,
)


def test_normalize_hf_batch_response_accepts_ragged_token_level_features():
    payload = [
        [[1.0] * 768, [9.0] * 768, [7.0] * 768],
        [[2.0] * 768],
    ]

    out = normalize_hf_batch_response(payload, expected_items=2)

    assert out.shape == (2, 768)
    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0], dtype=np.float32))


def test_request_hf_batch_async_uses_array_inputs_for_multi_text_batch():
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [
                [[1.0] * 768, [9.0] * 768],
                [[2.0] * 768],
            ]

    class _FakeClient:
        async def post(self, url, json):
            seen["url"] = url
            seen["json"] = json
            return _FakeResponse()

    vectors, meta = asyncio.run(
        _request_hf_batch_async(
            client=_FakeClient(),
            url="https://router.huggingface.co/hf-inference/models/allenai/specter/pipeline/feature-extraction",
            texts=["alpha", "beta"],
            max_retries=0,
            base_backoff_seconds=0.1,
            max_backoff_seconds=1.0,
        )
    )

    assert seen["json"] == {
        "inputs": ["alpha", "beta"],
        "parameters": {"truncate": True, "truncation_direction": "Right"},
    }
    assert vectors.shape == (2, 768)
    np.testing.assert_allclose(vectors[:, 0], np.array([1.0, 2.0], dtype=np.float32))
    assert meta["batch_size"] == 2

