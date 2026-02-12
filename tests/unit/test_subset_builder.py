import pandas as pd

from src.common.subset_builder import build_stage_subset


def _toy_mentions(n=200):
    rows = []
    for i in range(n):
        block = f"a.block{i%20}"
        rows.append(
            {
                "mention_id": f"b{i}::0",
                "bibcode": f"b{i}",
                "author_idx": 0,
                "author_raw": f"Author {i}",
                "title": "t",
                "abstract": "a",
                "year": 2000,
                "source_type": "toy",
                "block_key": block,
            }
        )
    return pd.DataFrame(rows)


def test_subset_deterministic_same_seed():
    m = _toy_mentions()
    s1 = build_stage_subset(m, stage="mini", seed=11, target_mentions=50)
    s2 = build_stage_subset(m, stage="mini", seed=11, target_mentions=50)
    assert s1["mention_id"].tolist() == s2["mention_id"].tolist()
