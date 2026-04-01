from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from author_name_disambiguation.infer_sources import InferSourcesRequest, run_infer_sources


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    pd.DataFrame(
        [
            {
                "Bibcode": "bib1",
                "Author": ["Doe J", "Roe A"],
                "Title_en": "Paper 1",
                "Abstract_en": "Abstract 1",
                "Year": 2020,
                "Affiliation": ["Inst A", "Inst B"],
            },
            {
                "Bibcode": "bib2",
                "Author": ["Doe J"],
                "Title_en": "Paper 2",
                "Abstract_en": "Abstract 2",
                "Year": 2021,
                "Affiliation": ["Inst C"],
            },
        ]
    ).to_parquet(publications_path, index=False)
    pd.DataFrame(
        [
            {
                "Bibcode": "bib3",
                "Author": ["Ref X"],
                "Title_en": "Paper 3",
                "Abstract_en": "Abstract 3",
                "Year": 2022,
                "Affiliation": ["Inst R"],
            }
        ]
    ).to_parquet(references_path, index=False)
    return publications_path, references_path


def _write_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "checkpoint.pt").write_text("checkpoint", encoding="utf-8")
    _write_yaml(
        bundle_dir / "model_config.yaml",
        {
            "name": "mock-nand",
            "representation": {"text_model_name": "allenai/specter", "max_length": 256},
            "training": {"precision_mode": "fp32"},
        },
    )
    with (bundle_dir / "clustering_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump({"eps_resolution": {"selected_eps": 0.35}}, handle, indent=2)
    with (bundle_dir / "bundle_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "bundle_schema_version": "v1",
                "source_model_run_id": "train_full_2026",
                "selected_eps": 0.35,
                "best_threshold": 0.25,
                "precision_mode": "fp32",
                "max_pairs_per_block": 100,
                "pair_building": {"exclude_same_bibcode": True},
            },
            handle,
            indent=2,
        )
    return bundle_dir


def _apply_pipeline_mocks(monkeypatch, *, text_mock):
    def _chars(mentions, output_path, **_kwargs):
        arr = np.ones((len(mentions), 50), dtype=np.float32)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "chars2vec",
            "name_count": int(len(mentions)),
            "unique_name_count": int(len(mentions)),
            "wall_seconds": 0.01,
            "generation_seconds": 0.01,
            "model_load_seconds": 0.0,
            "normalize_seconds": 0.0,
            "unique_seconds": 0.0,
            "pad_seconds": 0.0,
            "predict_seconds": 0.0,
            "materialize_seconds": 0.0,
            "cache_load_seconds": 0.0,
            "cache_write_seconds": 0.0,
            "tensorflow_memory_growth_enabled": True,
            "tensorflow_memory_growth_error": None,
            "tensorflow_cleanup_attempted": True,
            "tensorflow_cleanup_error": None,
        }
        return (arr, meta) if _kwargs.get("return_meta") else arr

    def _pairs(mentions, **_kwargs):
        rows = []
        for block_key, grp in mentions.groupby("block_key", sort=False):
            grp = grp.reset_index(drop=True)
            if len(grp) < 2:
                continue
            rows.append(
                {
                    "pair_id": f"{grp.loc[0, 'mention_id']}__{grp.loc[1, 'mention_id']}",
                    "mention_id_1": str(grp.loc[0, "mention_id"]),
                    "mention_id_2": str(grp.loc[1, "mention_id"]),
                    "block_key": str(block_key),
                    "split": "inference",
                    "label": None,
                }
            )
        out = pd.DataFrame(rows)
        meta = {
            "pairs_written": int(len(out)),
            "cpu_sharding_mode": "auto",
            "cpu_sharding_enabled": True,
            "cpu_workers_requested": "auto",
            "cpu_workers_effective": 2,
            "cpu_limit_detected": 4,
            "cpu_limit_source": "test",
            "cpu_min_pairs_per_worker": 1_000_000,
            "ram_budget_bytes": None,
            "total_pairs_est": int(len(out)),
            "group_blocks_seconds": 0.0,
            "partition_shards_seconds": 0.0,
            "oversize_sequential_seconds": 0.0,
            "worker_submit_seconds": 0.0,
            "worker_collect_seconds": 0.0,
            "merge_shards_seconds": 0.0,
            "final_readback_seconds": 0.0,
            "worker_compute_seconds_total": 0.0,
            "worker_flush_seconds_total": 0.0,
            "top_slow_blocks": [],
            "block_size_histogram": {"2": 1},
        }
        return (out, meta) if _kwargs.get("return_meta") else out

    def _score(mentions, pairs, output_path=None, **_kwargs):
        del mentions
        pairs_df = pairs if isinstance(pairs, pd.DataFrame) else pd.read_parquet(pairs)
        out = pairs_df[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
        out["cosine_sim"] = 0.9
        out["distance"] = 0.1
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        meta = {
            "requested_device": str(_kwargs.get("device", "cpu")),
            "resolved_device": str(_kwargs.get("device", "cpu")),
            "fallback_reason": None,
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "cuda_probe_error": None,
            "model_to_cuda_error": None,
            "effective_precision_mode": "fp32",
            "pair_scoring_strategy": "preencoded_mentions",
            "mention_storage_device": "cpu",
            "cuda_oom_fallback_used": False,
            "numeric_clamping": {"clamped": False},
        }
        return (out, meta) if _kwargs.get("return_runtime_meta") else out

    def _cluster(mentions, pair_scores, output_path=None, **_kwargs):
        del pair_scores
        out = mentions[["mention_id", "block_key"]].copy()
        out["author_uid"] = [f"uid.{idx}" for idx in range(len(out))]
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        meta = {
            "cluster_backend_requested": str(_kwargs.get("backend", "auto")),
            "cluster_backend_effective": "sklearn_cpu",
            "cluster_backend_reason": "test",
            "cpu_sharding_mode": "auto",
            "cpu_sharding_enabled": True,
            "cpu_workers_requested": "auto",
            "cpu_workers_effective": 2,
            "ram_budget_bytes": None,
            "total_pairs_est": int(len(out)),
            "block_p95": 2.0,
            "block_size_histogram": {"1": 1},
            "block_count_by_bucket": {"1": 1},
            "total_seconds_by_bucket": {"1": 0.001},
            "dbscan_seconds_by_bucket": {"1": 0.001},
            "distance_matrix_seconds_total": 0.0,
            "constraints_seconds_total": 0.0,
            "sanitize_seconds_total": 0.0,
            "dbscan_seconds_total": 0.0,
            "gpu_transfer_seconds_total": 0.0,
            "top_slow_blocks": [],
        }
        return (out, meta) if _kwargs.get("return_meta") else out

    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_specter_embeddings", text_mock)
    monkeypatch.setattr("author_name_disambiguation.source_inference.build_pairs_within_blocks", _pairs)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_with_checkpoint", _score)
    monkeypatch.setattr("author_name_disambiguation.source_inference.cluster_blockwise_dbscan", _cluster)
    monkeypatch.setattr(
        "author_name_disambiguation.source_inference._probe_bootstrap_runtime",
        lambda _device: {
            "requested_device": str(_device),
            "resolved_device": str(_device),
            "gpu_name": None,
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "fallback_reason": None,
            "cuda_probe_error": None,
        },
    )


def test_runtime_mode_cpu_falls_back_from_onnx_to_transformers(monkeypatch, tmp_path: Path):
    publications_path, references_path = _write_dataset(tmp_path)
    bundle_dir = _write_bundle(tmp_path)
    calls: list[str] = []

    def _text(mentions, output_path, **kwargs):
        runtime_backend = str(kwargs["runtime_backend"])
        calls.append(runtime_backend)
        if runtime_backend == "onnx_fp32":
            raise RuntimeError("onnx init failed")
        arr = np.ones((len(mentions), 768), dtype=np.float32)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "model_only",
            "runtime_backend": runtime_backend,
            "requested_device": str(kwargs.get("device", "cpu")),
            "resolved_device": "cpu",
            "fallback_reason": None,
            "effective_precision_mode": "fp32",
            "requested_batch_size": kwargs.get("batch_size"),
            "effective_batch_size": kwargs.get("batch_size"),
            "column_present": False,
            "precomputed_embedding_count": 0,
            "recomputed_embedding_count": int(len(mentions)),
            "used_precomputed_embeddings": False,
        }
        return arr, meta

    _apply_pipeline_mocks(monkeypatch, text_mock=_text)
    result = run_infer_sources(
        InferSourcesRequest(
            publications_path=publications_path,
            references_path=references_path,
            output_root=tmp_path / "out_cpu",
            dataset_id="ads_runtime_cpu",
            model_bundle=bundle_dir,
            runtime_mode="cpu",
            progress=False,
        )
    )

    stage_metrics = json.loads(result.stage_metrics_path.read_text(encoding="utf-8"))
    assert calls == ["onnx_fp32", "transformers"]
    assert stage_metrics["runtime"]["specter"]["runtime_mode"] == "cpu"
    assert stage_metrics["runtime"]["specter"]["runtime_backend"] == "transformers"
    assert stage_metrics["runtime"]["specter"]["fallback_reason"] == "cpu_auto_onnx_fallback"


def test_runtime_mode_hf_uses_direct_hf_backend(monkeypatch, tmp_path: Path):
    publications_path, references_path = _write_dataset(tmp_path)
    bundle_dir = _write_bundle(tmp_path)
    calls: list[str] = []

    def _text(mentions, output_path, **kwargs):
        runtime_backend = str(kwargs["runtime_backend"])
        calls.append(runtime_backend)
        arr = np.ones((len(mentions), 768), dtype=np.float32)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "remote_endpoint_only",
            "runtime_backend": runtime_backend,
            "requested_device": "hf",
            "resolved_device": "remote:hf-endpoint",
            "fallback_reason": None,
            "effective_precision_mode": None,
            "api_concurrency": 8,
            "request_batch_size": 64,
            "column_present": False,
            "precomputed_embedding_count": 0,
            "recomputed_embedding_count": int(len(mentions)),
            "used_precomputed_embeddings": False,
        }
        return arr, meta

    _apply_pipeline_mocks(monkeypatch, text_mock=_text)
    result = run_infer_sources(
        InferSourcesRequest(
            publications_path=publications_path,
            references_path=references_path,
            output_root=tmp_path / "out_hf",
            dataset_id="ads_runtime_hf",
            model_bundle=bundle_dir,
            runtime_mode="hf",
            progress=False,
        )
    )

    context = json.loads((result.output_root / "00_context.json").read_text(encoding="utf-8"))
    stage_metrics = json.loads(result.stage_metrics_path.read_text(encoding="utf-8"))
    assert calls == ["hf_endpoint"]
    assert context["runtime_mode"] == "hf"
    assert context["runtime_backend"] == "hf_endpoint"
    assert context["resolved_device"] == "remote:hf-endpoint"
    assert context["generation_mode"] == "remote_endpoint_only"
    assert stage_metrics["runtime"]["specter"]["runtime_mode"] == "hf"
    assert stage_metrics["runtime"]["specter"]["runtime_backend"] == "hf_endpoint"
    assert stage_metrics["runtime"]["specter"]["resolved_device"] == "remote:hf-endpoint"


def test_internal_backend_override_keeps_public_runtime_metadata_compact(monkeypatch, tmp_path: Path):
    publications_path, references_path = _write_dataset(tmp_path)
    bundle_dir = _write_bundle(tmp_path)

    def _text(mentions, output_path, **kwargs):
        arr = np.ones((len(mentions), 768), dtype=np.float32)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "model_only",
            "runtime_backend": str(kwargs["runtime_backend"]),
            "requested_device": str(kwargs.get("device", "cpu")),
            "resolved_device": "cpu",
            "fallback_reason": None,
            "effective_precision_mode": "fp32",
            "column_present": False,
            "precomputed_embedding_count": 0,
            "recomputed_embedding_count": int(len(mentions)),
            "used_precomputed_embeddings": False,
        }
        return arr, meta

    _apply_pipeline_mocks(monkeypatch, text_mock=_text)
    result = run_infer_sources(
        InferSourcesRequest(
            publications_path=publications_path,
            references_path=references_path,
            output_root=tmp_path / "out_legacy",
            dataset_id="ads_runtime_legacy",
            model_bundle=bundle_dir,
            device="cpu",
            specter_runtime_backend="transformers",
            progress=False,
        )
    )

    context = json.loads((result.output_root / "00_context.json").read_text(encoding="utf-8"))
    stage_metrics = json.loads(result.stage_metrics_path.read_text(encoding="utf-8"))

    assert context["runtime_mode"] == "cpu"
    assert context["runtime_backend"] == "transformers"
    assert context["resolved_device"] == "cpu"
    assert context["generation_mode"] == "model_only"
    assert "legacy_runtime_overrides" not in context
    assert "legacy_runtime_overrides" not in stage_metrics["runtime"]["specter"]
