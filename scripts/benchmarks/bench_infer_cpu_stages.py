#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.approaches.nand.build_pairs import build_pairs_within_blocks
from src.approaches.nand.cluster import cluster_blockwise_dbscan
from src.common.cpu_runtime import (
    compute_ram_budget_bytes,
    detect_available_ram_bytes,
    detect_cpu_limit,
    normalize_workers_request,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_output_path(repo_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "artifacts" / "benchmarks" / f"cpu_sharding_{ts}.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _prepare_cluster_config(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return {
            "eps": 0.35,
            "min_samples": 1,
            "metric": "precomputed",
            "constraints": {"enabled": False},
        }, "inline_default"
    cfg = _load_yaml(path)
    return cfg, str(path)


def _to_pair_scores(
    *,
    pairs: pd.DataFrame,
    seed: int,
    external_pair_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    if external_pair_scores is not None:
        required = {"pair_id", "mention_id_1", "mention_id_2", "block_key"}
        missing = sorted(list(required - set(external_pair_scores.columns)))
        if missing:
            raise ValueError(f"pair_scores_path is missing required columns: {missing}")
        out = external_pair_scores.copy()
        if "distance" not in out.columns:
            if "cosine_sim" in out.columns:
                out["distance"] = 1.0 - out["cosine_sim"].astype(np.float32)
            else:
                raise ValueError("pair_scores_path must include either distance or cosine_sim.")
        return out[["pair_id", "mention_id_1", "mention_id_2", "block_key", "distance"]].copy()

    out = pairs[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
    if len(out) == 0:
        out["distance"] = np.array([], dtype=np.float32)
        return out

    rng = np.random.default_rng(seed)
    out["distance"] = rng.uniform(0.05, 0.35, size=len(out)).astype(np.float32)
    return out


def _median_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(statistics.median(values))


def _safe_speedup(base: float, candidate: float) -> float | None:
    if base <= 0.0 or candidate <= 0.0:
        return None
    return float(base / candidate)


def _amdahl_projection(cpu_speedup: float, cpu_share: float) -> float:
    if cpu_speedup <= 0.0:
        return 1.0
    return float(1.0 / ((1.0 - cpu_share) + cpu_share / cpu_speedup))


def _run_case_once(
    *,
    case_name: str,
    mentions: pd.DataFrame,
    pairs_external: pd.DataFrame | None,
    pair_scores_external: pd.DataFrame | None,
    cluster_config: dict[str, Any],
    build_seed: int,
    score_seed: int,
    max_pairs_per_block: int | None,
    cpu_min_pairs_per_worker: int,
    ram_budget_bytes: int | None,
    pair_num_workers: int | None,
    pair_sharding_mode: str,
    cluster_num_workers: int | None,
    cluster_sharding_mode: str,
    cluster_backend: str,
) -> dict[str, Any]:
    pair_t0 = time.perf_counter()
    if pairs_external is None:
        pairs, pair_meta = build_pairs_within_blocks(
            mentions=mentions,
            max_pairs_per_block=max_pairs_per_block,
            seed=int(build_seed),
            require_same_split=False,
            labeled_only=False,
            balance_train=False,
            exclude_same_bibcode=True,
            show_progress=False,
            output_path=None,
            return_pairs=True,
            return_meta=True,
            num_workers=pair_num_workers,
            sharding_mode=pair_sharding_mode,
            min_pairs_per_worker=int(cpu_min_pairs_per_worker),
            ram_budget_bytes=ram_budget_bytes,
        )
        pair_stage_mode = "build"
    else:
        pairs = pairs_external.copy()
        pair_meta = {
            "cpu_workers_effective": 0,
            "cpu_sharding_enabled": False,
            "cpu_sharding_mode": "external_pairs",
        }
        pair_stage_mode = "load_external"
    pair_t1 = time.perf_counter()

    pair_scores = _to_pair_scores(
        pairs=pairs,
        seed=int(score_seed),
        external_pair_scores=pair_scores_external,
    )

    cluster_t0 = time.perf_counter()
    clusters, cluster_meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cluster_config,
        output_path=None,
        show_progress=False,
        num_workers=cluster_num_workers,
        sharding_mode=cluster_sharding_mode,
        min_pairs_per_worker=int(cpu_min_pairs_per_worker),
        ram_budget_bytes=ram_budget_bytes,
        backend=cluster_backend,
        return_meta=True,
    )
    cluster_t1 = time.perf_counter()

    return {
        "case": case_name,
        "pair_stage_mode": pair_stage_mode,
        "pair_seconds": round(pair_t1 - pair_t0, 6),
        "cluster_seconds": round(cluster_t1 - cluster_t0, 6),
        "total_cpu_seconds": round((pair_t1 - pair_t0) + (cluster_t1 - cluster_t0), 6),
        "pairs": int(len(pairs)),
        "clusters": int(len(clusters)),
        "pair_workers_effective": int(pair_meta.get("cpu_workers_effective", 0)),
        "cluster_workers_effective": int(cluster_meta.get("cpu_workers_effective", 0)),
        "pair_sharding_enabled": bool(pair_meta.get("cpu_sharding_enabled", False)),
        "cluster_sharding_enabled": bool(cluster_meta.get("cpu_sharding_enabled", False)),
        "cluster_backend_effective": str(cluster_meta.get("cluster_backend_effective", cluster_backend)),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark NAND CPU-heavy infer stages.")
    p.add_argument("--mentions-path", required=True, help="Parquet path with ADS mentions.")
    p.add_argument("--pairs-path", default=None, help="Optional external pairs parquet. If set, pair build is skipped.")
    p.add_argument(
        "--pair-scores-path",
        default=None,
        help="Optional external pair_scores parquet (must include distance or cosine_sim).",
    )
    p.add_argument("--cluster-config", default=None, help="Optional clustering YAML config path.")
    p.add_argument("--output", default=None, help="Output JSON path. Default: artifacts/benchmarks/cpu_sharding_<ts>.json")

    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measure-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=11)

    p.add_argument("--max-pairs-per-block", type=int, default=None)
    p.add_argument("--cpu-min-pairs-per-worker", type=int, default=1_000_000)
    p.add_argument("--cpu-target-ram-fraction", type=float, default=0.6)

    p.add_argument("--baseline-workers", default="1")
    p.add_argument("--baseline-sharding", choices=["auto", "on", "off"], default="off")
    p.add_argument("--optimized-workers", default="auto")
    p.add_argument("--optimized-sharding", choices=["auto", "on", "off"], default="on")
    p.add_argument("--cluster-backend", choices=["auto", "sklearn_cpu", "cuml_gpu"], default="auto")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    mentions_path = Path(args.mentions_path)
    if not mentions_path.exists():
        raise FileNotFoundError(mentions_path)
    mentions = pd.read_parquet(mentions_path)

    pairs_external = None
    if args.pairs_path:
        pairs_path = Path(args.pairs_path)
        if not pairs_path.exists():
            raise FileNotFoundError(pairs_path)
        pairs_external = pd.read_parquet(pairs_path)

    pair_scores_external = None
    if args.pair_scores_path:
        pair_scores_path = Path(args.pair_scores_path)
        if not pair_scores_path.exists():
            raise FileNotFoundError(pair_scores_path)
        pair_scores_external = pd.read_parquet(pair_scores_path)

    cluster_config_path = Path(args.cluster_config) if args.cluster_config else None
    cluster_config, cluster_config_source = _prepare_cluster_config(cluster_config_path)

    if int(args.warmup_runs) < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if int(args.measure_runs) < 1:
        raise ValueError("--measure-runs must be >= 1")
    if int(args.cpu_min_pairs_per_worker) < 1:
        raise ValueError("--cpu-min-pairs-per-worker must be >= 1")
    if not (0.0 < float(args.cpu_target_ram_fraction) <= 1.0):
        raise ValueError("--cpu-target-ram-fraction must be in (0, 1].")

    baseline_workers = normalize_workers_request(args.baseline_workers)
    optimized_workers = normalize_workers_request(args.optimized_workers)

    available_ram_bytes = detect_available_ram_bytes()
    ram_budget_bytes = compute_ram_budget_bytes(
        target_fraction=float(args.cpu_target_ram_fraction),
        available_ram_bytes=available_ram_bytes,
    )
    cpu_info = detect_cpu_limit()

    cases = [
        {
            "name": "baseline_seq",
            "pair_num_workers": baseline_workers,
            "pair_sharding_mode": str(args.baseline_sharding),
            "cluster_num_workers": baseline_workers,
            "cluster_sharding_mode": str(args.baseline_sharding),
        },
        {
            "name": "optimized",
            "pair_num_workers": optimized_workers,
            "pair_sharding_mode": str(args.optimized_sharding),
            "cluster_num_workers": optimized_workers,
            "cluster_sharding_mode": str(args.optimized_sharding),
        },
    ]

    report_cases: dict[str, Any] = {}
    for case in cases:
        warmup_rows: list[dict[str, Any]] = []
        measured_rows: list[dict[str, Any]] = []
        for _ in range(int(args.warmup_runs)):
            warmup_rows.append(
                _run_case_once(
                    case_name=case["name"],
                    mentions=mentions,
                    pairs_external=pairs_external,
                    pair_scores_external=pair_scores_external,
                    cluster_config=cluster_config,
                    build_seed=int(args.seed),
                    score_seed=int(args.seed),
                    max_pairs_per_block=args.max_pairs_per_block,
                    cpu_min_pairs_per_worker=int(args.cpu_min_pairs_per_worker),
                    ram_budget_bytes=ram_budget_bytes,
                    pair_num_workers=case["pair_num_workers"],
                    pair_sharding_mode=case["pair_sharding_mode"],
                    cluster_num_workers=case["cluster_num_workers"],
                    cluster_sharding_mode=case["cluster_sharding_mode"],
                    cluster_backend=str(args.cluster_backend),
                )
            )

        for run_idx in range(int(args.measure_runs)):
            measured_rows.append(
                _run_case_once(
                    case_name=case["name"],
                    mentions=mentions,
                    pairs_external=pairs_external,
                    pair_scores_external=pair_scores_external,
                    cluster_config=cluster_config,
                    build_seed=int(args.seed + run_idx + 1),
                    score_seed=int(args.seed + run_idx + 1),
                    max_pairs_per_block=args.max_pairs_per_block,
                    cpu_min_pairs_per_worker=int(args.cpu_min_pairs_per_worker),
                    ram_budget_bytes=ram_budget_bytes,
                    pair_num_workers=case["pair_num_workers"],
                    pair_sharding_mode=case["pair_sharding_mode"],
                    cluster_num_workers=case["cluster_num_workers"],
                    cluster_sharding_mode=case["cluster_sharding_mode"],
                    cluster_backend=str(args.cluster_backend),
                )
            )

        report_cases[case["name"]] = {
            "warmup_runs": warmup_rows,
            "measured_runs": measured_rows,
            "median": {
                "pair_seconds": round(_median_metric(measured_rows, "pair_seconds"), 6),
                "cluster_seconds": round(_median_metric(measured_rows, "cluster_seconds"), 6),
                "total_cpu_seconds": round(_median_metric(measured_rows, "total_cpu_seconds"), 6),
                "pair_workers_effective": int(statistics.median([r["pair_workers_effective"] for r in measured_rows])),
                "cluster_workers_effective": int(
                    statistics.median([r["cluster_workers_effective"] for r in measured_rows])
                ),
                "pair_sharding_enabled_any": bool(any(r["pair_sharding_enabled"] for r in measured_rows)),
                "cluster_sharding_enabled_any": bool(any(r["cluster_sharding_enabled"] for r in measured_rows)),
                "cluster_backend_effective": str(measured_rows[-1]["cluster_backend_effective"]),
            },
        }

    baseline = report_cases["baseline_seq"]["median"]
    optimized = report_cases["optimized"]["median"]
    speedup_pair = _safe_speedup(float(baseline["pair_seconds"]), float(optimized["pair_seconds"]))
    speedup_cluster = _safe_speedup(float(baseline["cluster_seconds"]), float(optimized["cluster_seconds"]))
    speedup_total = _safe_speedup(float(baseline["total_cpu_seconds"]), float(optimized["total_cpu_seconds"]))

    gates = {
        "speedup_pair": None if speedup_pair is None else round(speedup_pair, 6),
        "speedup_cluster": None if speedup_cluster is None else round(speedup_cluster, 6),
        "speedup_total_cpu": None if speedup_total is None else round(speedup_total, 6),
        "target_met_pair_2x": bool(speedup_pair is not None and speedup_pair >= 2.0),
        "target_met_cluster_2x": bool(speedup_cluster is not None and speedup_cluster >= 2.0),
        "target_met_total_2x": bool(speedup_total is not None and speedup_total >= 2.0),
    }

    amdahl = {}
    if speedup_total is not None:
        for share in (0.4, 0.6, 0.8):
            amdahl[f"cpu_share_{share:.1f}"] = round(_amdahl_projection(speedup_total, share), 6)

    out_payload = {
        "generated_utc": _utc_now(),
        "mentions_path": str(mentions_path),
        "mentions_count": int(len(mentions)),
        "cluster_config_source": cluster_config_source,
        "benchmark_config": {
            "warmup_runs": int(args.warmup_runs),
            "measure_runs": int(args.measure_runs),
            "seed": int(args.seed),
            "max_pairs_per_block": args.max_pairs_per_block,
            "cpu_min_pairs_per_worker": int(args.cpu_min_pairs_per_worker),
            "cpu_target_ram_fraction": float(args.cpu_target_ram_fraction),
            "baseline_workers": "auto" if baseline_workers is None else int(baseline_workers),
            "baseline_sharding": str(args.baseline_sharding),
            "optimized_workers": "auto" if optimized_workers is None else int(optimized_workers),
            "optimized_sharding": str(args.optimized_sharding),
            "cluster_backend_requested": str(args.cluster_backend),
            "pairs_path": None if args.pairs_path is None else str(Path(args.pairs_path)),
            "pair_scores_path": None if args.pair_scores_path is None else str(Path(args.pair_scores_path)),
        },
        "environment": {
            "cpu_limit_info": cpu_info,
            "available_ram_bytes": None if available_ram_bytes is None else int(available_ram_bytes),
            "ram_budget_bytes": None if ram_budget_bytes is None else int(ram_budget_bytes),
        },
        "cases": report_cases,
        "gates": gates,
        "amdahl_projection": amdahl,
    }

    output_path = Path(args.output) if args.output else _default_output_path(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(json.dumps({"output_path": str(output_path), "gates": gates, "amdahl_projection": amdahl}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
