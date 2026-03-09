import pandas as pd

from author_name_disambiguation.common.io_schema import validate_pair_score_ranges


def test_validate_pair_score_ranges_reports_out_of_range_values():
    df = pd.DataFrame(
        [
            {"cosine_sim": 1.0000001, "distance": -1e-7},
            {"cosine_sim": -1.1, "distance": 2.1},
            {"cosine_sim": None, "distance": None},
        ]
    )

    stats = validate_pair_score_ranges(df)

    assert stats["pair_score_range_ok"] is False
    assert stats["cosine_non_finite_count"] == 1
    assert stats["distance_non_finite_count"] == 1
    assert stats["cosine_out_of_range_count"] == 2
    assert stats["negative_distance_count"] == 1
    assert stats["distance_above_max_count"] == 1
