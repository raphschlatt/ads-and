from __future__ import annotations

import numpy as np
import pandas as pd

from author_name_disambiguation.source_inference import (
    _fanout_specter_embeddings_to_mentions,
    _select_specter_source_records,
)


def test_select_specter_source_records_keeps_canonical_order():
    canonical_records = pd.DataFrame(
        {
            "canonical_record_id": [0, 1, 2],
            "bibcode": ["bib1", "bib2", "bib3"],
            "title": ["Paper 1", "Paper 2", "Paper 3"],
            "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
        }
    )
    mentions = pd.DataFrame(
        {
            "mention_id": ["bib3::0", "bib1::0", "bib1::1"],
            "canonical_record_id": [2, 0, 0],
        }
    )

    out = _select_specter_source_records(canonical_records=canonical_records, mentions=mentions)

    assert out["canonical_record_id"].tolist() == [0, 2]
    assert out["bibcode"].tolist() == ["bib1", "bib3"]


def test_fanout_specter_embeddings_reuses_one_source_vector_per_canonical_record():
    specter_source_records = pd.DataFrame({"canonical_record_id": [10, 20, 30]})
    mentions = pd.DataFrame(
        {
            "mention_id": ["a::0", "a::1", "b::0", "c::0", "c::1"],
            "canonical_record_id": [10, 10, 20, 30, 30],
        }
    )
    source_embeddings = np.asarray(
        [
            np.full(4, 1.0, dtype=np.float32),
            np.full(4, 2.0, dtype=np.float32),
            np.full(4, 3.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    out = _fanout_specter_embeddings_to_mentions(
        specter_source_records=specter_source_records,
        mentions=mentions,
        source_embeddings=source_embeddings,
    )

    assert out.shape == (5, 4)
    np.testing.assert_allclose(out[0], source_embeddings[0])
    np.testing.assert_allclose(out[1], source_embeddings[0])
    np.testing.assert_allclose(out[2], source_embeddings[1])
    np.testing.assert_allclose(out[3], source_embeddings[2])
    np.testing.assert_allclose(out[4], source_embeddings[2])
