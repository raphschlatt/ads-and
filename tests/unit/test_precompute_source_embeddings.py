from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from author_name_disambiguation.embedding_contract import build_source_text
from author_name_disambiguation.hf_transport import normalize_hf_batch_response
from author_name_disambiguation.precompute_source_embeddings import _resolve_hf_token


def test_build_source_text_matches_local_specter_path():
    assert build_source_text("A title", "An abstract") == "A title [SEP] An abstract"
    assert build_source_text("A title", "") == "A title"
    assert build_source_text("", "An abstract") == "An abstract"


def test_normalize_hf_batch_response_accepts_document_vectors():
    payload = [[0.1] * 768, [0.2] * 768]
    out = normalize_hf_batch_response(payload, expected_items=2)
    assert out.shape == (2, 768)
    np.testing.assert_allclose(out[0, :3], np.array([0.1, 0.1, 0.1], dtype=np.float32))
    np.testing.assert_allclose(out[1, :3], np.array([0.2, 0.2, 0.2], dtype=np.float32))


def test_normalize_hf_batch_response_accepts_token_level_features_and_uses_first_token():
    payload = [
        [[1.0] * 768, [9.0] * 768],
        [[2.0] * 768, [8.0] * 768],
    ]
    out = normalize_hf_batch_response(payload, expected_items=2)
    assert out.shape == (2, 768)
    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0], dtype=np.float32))


def test_normalize_hf_batch_response_rejects_bad_shape():
    with pytest.raises(ValueError, match="Incompatible HF batch response shape"):
        normalize_hf_batch_response([[1.0] * 32], expected_items=2)


def test_resolve_hf_token_reads_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    assert _resolve_hf_token("HF_TOKEN") == "secret-token"


def test_resolve_hf_token_raises_when_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Missing Hugging Face token"):
        _resolve_hf_token("HF_TOKEN")


def test_precompute_uses_shared_endpoint_helper(monkeypatch: pytest.MonkeyPatch):
    precompute_module = importlib.import_module("author_name_disambiguation.precompute_source_embeddings")
    seen: dict[str, object] = {}

    def _fake_embed_texts_via_hf_endpoint(**kwargs):
        seen.update(kwargs)
        return np.full((1, 768), 0.5, dtype=np.float32), {"transport": "hf_endpoint", "attempts_total": 2}

    monkeypatch.setattr(precompute_module, "embed_texts_via_hf_endpoint", _fake_embed_texts_via_hf_endpoint)

    source = precompute_module._LoadedSource(
        label="publications",
        input_path=precompute_module.Path("publications.parquet"),
        normalized=pd.DataFrame(
            [
                {
                    "bibcode": "bib1",
                    "authors": ["Doe J"],
                    "title": "Paper 1",
                    "abstract": "Abstract 1",
                    "year": 2020,
                    "aff": ["Inst A"],
                }
            ]
        ),
        raw_source=None,
        output_path=precompute_module.Path("publications_precomputed.parquet"),
        load_meta={},
    )

    vectors, meta = precompute_module._compute_vectors_for_source_records(
        sources=[source],
        request=precompute_module.PrecomputeSourceEmbeddingsRequest(
            publications_path="pubs.parquet",
            output_root="out",
            progress=False,
        ),
    )

    assert list(vectors) == ["publications"]
    assert meta["hf_runtime"]["transport"] == "hf_endpoint"
    assert seen["texts"] == ["Paper 1 [SEP] Abstract 1"]
