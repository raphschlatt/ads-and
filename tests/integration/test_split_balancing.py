import pandas as pd

from src.approaches.nand.build_pairs import assign_lspo_splits


def _mentions(n_orcid: int, reps_per_orcid: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_orcid):
        for j in range(reps_per_orcid):
            rows.append(
                {
                    "mention_id": f"b{i}_{j}::0",
                    "bibcode": f"b{i}_{j}",
                    "author_idx": 0,
                    "author_raw": f"Smith, A{i}",
                    "title": "t",
                    "abstract": "a",
                    "year": 2000 + j,
                    "source_type": "lspo",
                    "block_key": "a.smith",
                    "orcid": f"o{i}",
                }
            )
    return pd.DataFrame(rows)


def test_assign_lspo_splits_reaches_negative_targets_when_possible():
    df = _mentions(n_orcid=20, reps_per_orcid=2)
    split_df, meta = assign_lspo_splits(
        df,
        seed=11,
        train_ratio=0.4,
        val_ratio=0.3,
        min_neg_val=5,
        min_neg_test=5,
        max_attempts=20,
        return_meta=True,
    )

    assert len(split_df) == len(df)
    assert meta["status"] == "ok"
    assert meta["split_label_counts"]["val"]["neg"] >= 5
    assert meta["split_label_counts"]["test"]["neg"] >= 5


def test_assign_lspo_splits_marks_degraded_when_targets_impossible():
    # With 10 ORCIDs and val_ratio=0.1, val/test usually only receive one ORCID each.
    # Then negatives in val/test are structurally impossible.
    df = _mentions(n_orcid=10, reps_per_orcid=2)
    _, meta = assign_lspo_splits(
        df,
        seed=11,
        train_ratio=0.8,
        val_ratio=0.1,
        min_neg_val=1,
        min_neg_test=1,
        max_attempts=10,
        return_meta=True,
    )

    assert meta["status"] == "split_balance_degraded"
