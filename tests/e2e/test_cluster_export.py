import pandas as pd

from src.approaches.nand.cluster import cluster_blockwise_dbscan
from src.approaches.nand.export import build_publication_author_mapping


def test_cluster_then_export_consistency():
    mentions = pd.DataFrame(
        [
            {"mention_id": "b1::0", "bibcode": "b1", "author_idx": 0, "author_raw": "Doe, John", "title": "t", "abstract": "a", "year": 2000, "source_type": "ads", "block_key": "j.doe"},
            {"mention_id": "b2::0", "bibcode": "b2", "author_idx": 0, "author_raw": "Doe, John", "title": "t", "abstract": "a", "year": 2001, "source_type": "ads", "block_key": "j.doe"},
        ]
    )

    pair_scores = pd.DataFrame(
        [
            {
                "pair_id": "b1::0__b2::0",
                "mention_id_1": "b1::0",
                "mention_id_2": "b2::0",
                "block_key": "j.doe",
                "cosine_sim": 0.95,
                "distance": 0.05,
            }
        ]
    )

    cfg = {"eps": 0.35, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": True, "max_year_gap": 30, "enforce_name_conflict": True}}
    clusters = cluster_blockwise_dbscan(mentions=mentions, pair_scores=pair_scores, cluster_config=cfg)
    exported = build_publication_author_mapping(mentions=mentions, clusters=clusters)

    assert exported["mention_id"].nunique() == len(mentions)
    assert exported.groupby("mention_id")["author_uid"].nunique().max() == 1
