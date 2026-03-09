from __future__ import annotations

from pathlib import Path

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
