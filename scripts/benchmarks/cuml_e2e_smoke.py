#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import pandas as pd

from author_name_disambiguation.approaches.nand.cluster import cluster_blockwise_dbscan


def _toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2000},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2001},
        ]
    )
    pair_scores = pd.DataFrame(
        [
            {"pair_id": "a1__a2", "mention_id_1": "a1", "mention_id_2": "a2", "block_key": "blk.a", "distance": 0.05},
            {"pair_id": "b1__b2", "mention_id_1": "b1", "mention_id_2": "b2", "block_key": "blk.b", "distance": 0.05},
        ]
    )
    cluster_config = {"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}}
    return mentions, pair_scores, cluster_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test cuML clustering backend resolution and fallback behavior.")
    parser.add_argument(
        "--require-gpu-backend",
        action="store_true",
        help="Fail unless `cluster_backend=auto` resolves to `cuml_gpu`.",
    )
    args = parser.parse_args()

    mentions, pair_scores, cluster_config = _toy_inputs()

    # Case A: real runtime resolution path.
    _clusters_auto, meta_auto = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cluster_config,
        backend="auto",
        return_meta=True,
    )

    # Case B: deterministic GPU-failure fallback path.
    with patch(
        "src.approaches.nand.cluster._resolve_cluster_backend",
        lambda backend, metric: {
            "requested": str(backend),
            "effective": "cuml_gpu",
            "reason": "forced-test",
            "cuml_available": True,
            "metric": str(metric),
        },
    ), patch(
        "src.approaches.nand.cluster._run_dbscan_cuml",
        lambda dist, eps, min_samples, metric: (_ for _ in ()).throw(RuntimeError("forced gpu failure")),
    ):
        _clusters_fallback, meta_fallback = cluster_blockwise_dbscan(
            mentions=mentions,
            pair_scores=pair_scores,
            cluster_config=cluster_config,
            backend="auto",
            return_meta=True,
        )

    if args.require_gpu_backend and meta_auto.get("cluster_backend_effective") != "cuml_gpu":
        raise SystemExit(
            "Expected auto backend to resolve to cuml_gpu, "
            f"got {meta_auto.get('cluster_backend_effective')!r} "
            f"(reason={meta_auto.get('cluster_backend_reason')!r})."
        )

    if meta_fallback.get("cluster_backend_effective") != "sklearn_cpu":
        raise SystemExit(
            "Expected forced GPU failure fallback to resolve to sklearn_cpu, "
            f"got {meta_fallback.get('cluster_backend_effective')!r}."
        )

    print(
        json.dumps(
            {
                "auto_meta": meta_auto,
                "fallback_meta": meta_fallback,
                "require_gpu_backend": bool(args.require_gpu_backend),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
