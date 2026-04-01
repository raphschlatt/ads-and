from __future__ import annotations

import json
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from author_name_disambiguation import cli
from author_name_disambiguation.infer_sources import InferSourcesRequest, run_infer_sources


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


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
                "embedding_contract": {
                    "text": {
                        "family": "specter",
                        "provider": "huggingface",
                        "field_name": "precomputed_embedding",
                        "legacy_field_names": ["embedding"],
                        "model_name": "allenai/specter",
                        "text_backend": "transformers",
                        "text_adapter_name": None,
                        "text_adapter_alias": "specter2",
                        "dimension": 768,
                        "text_builder": "title [SEP] abstract",
                        "separator": " [SEP] ",
                        "title_columns": ["Title_en", "Title", "title"],
                        "abstract_columns": ["Abstract_en", "Abstract", "abstract"],
                        "pooling": "cls_first_token",
                        "tokenization": {"truncation": True, "max_length": 256},
                    },
                    "name": {"family": "chars2vec", "model_name": "eng_50", "dimension": 50},
                },
            },
            handle,
            indent=2,
        )
    return bundle_dir


def _apply_fast_infer_mocks(monkeypatch):
    def _chars(mentions, output_path, force_recompute=False, **_kwargs):
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

    def _text(mentions, output_path, force_recompute=False, **_kwargs):
        vectors = mentions["precomputed_embedding"].tolist()
        assert all(isinstance(row, list) and len(row) == 768 for row in vectors)
        arr = np.asarray(vectors, dtype=np.float32)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "precomputed_only",
            "requested_device": str(_kwargs.get("device", "auto")),
            "resolved_device": None,
            "fallback_reason": None,
            "torch_version": None,
            "torch_cuda_version": None,
            "torch_cuda_available": None,
            "cuda_probe_error": None,
            "model_to_cuda_error": None,
            "effective_precision_mode": "fp32",
            "requested_batch_size": _kwargs.get("batch_size"),
            "effective_batch_size": _kwargs.get("batch_size"),
            "oom_retry_count": 0,
            "batches_total": 0,
            "tokenize_seconds_total": 0.0,
            "host_to_device_seconds_total": 0.0,
            "forward_seconds_total": 0.0,
            "device_to_host_seconds_total": 0.0,
            "token_count_total": 0,
            "max_sequence_length_observed": 0,
            "mean_sequence_length_observed": 0.0,
            "device_to_host_flushes": 0,
            "column_present": True,
            "precomputed_embedding_count": int(len(mentions)),
            "recomputed_embedding_count": 0,
            "used_precomputed_embeddings": True,
        }
        return (arr, meta) if _kwargs.get("return_meta") else arr

    def _pairs(mentions, output_path=None, **_kwargs):
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
            "same_publication_pairs_skipped": 0,
            "pairs_written": int(len(out)),
            "chunk_rows": 0,
            "output_path": None if output_path is None else str(output_path),
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
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        return (out, meta) if _kwargs.get("return_meta") else out

    def _score(mentions, pairs, output_path=None, **_kwargs):
        del mentions
        pairs_df = pd.read_parquet(pairs) if isinstance(pairs, (str, Path)) else pairs
        out = pairs_df[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
        out["cosine_sim"] = 0.9
        out["distance"] = 0.1
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        meta = {
            "requested_device": "cpu",
            "resolved_device": "cpu",
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
            "cluster_backend_requested": "sklearn_cpu",
            "cluster_backend_effective": "sklearn_cpu",
            "cluster_backend_reason": "test",
            "cpu_sharding_mode": "auto",
            "cpu_sharding_enabled": True,
            "cpu_workers_requested": "auto",
            "cpu_workers_effective": 2,
            "ram_budget_bytes": None,
            "total_pairs_est": 1,
            "block_p95": 2.0,
            "block_size_histogram": {"1": 1},
            "block_count_by_bucket": {"1": 1},
            "total_seconds_by_bucket": {"1": 0.0},
            "dbscan_seconds_by_bucket": {"1": 0.0},
            "distance_matrix_seconds_total": 0.0,
            "constraints_seconds_total": 0.0,
            "sanitize_seconds_total": 0.0,
            "dbscan_seconds_total": 0.0,
            "gpu_transfer_seconds_total": 0.0,
            "top_slow_blocks": [],
        }
        return (out, meta) if _kwargs.get("return_meta") else out

    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_specter_embeddings", _text)
    monkeypatch.setattr("author_name_disambiguation.source_inference.build_pairs_within_blocks", _pairs)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_with_checkpoint", _score)
    monkeypatch.setattr("author_name_disambiguation.source_inference.cluster_blockwise_dbscan", _cluster)
    monkeypatch.setattr(
        "author_name_disambiguation.source_inference._probe_bootstrap_runtime",
        lambda _device: {
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "gpu_name": None,
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "fallback_reason": None,
            "cuda_probe_error": None,
        },
    )


def test_cli_precompute_source_embeddings_and_cpu_infer_use_precomputed(monkeypatch, tmp_path: Path):
    precompute_module = importlib.import_module("author_name_disambiguation.precompute_source_embeddings")
    publications_path = tmp_path / "publications.parquet"
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
                "precomputed_embedding": [0.25] * 768,
            },
        ]
    ).to_parquet(publications_path, index=False)

    def _fake_embed_texts_via_hf_endpoint(**kwargs):
        assert kwargs["model_name"] == "allenai/specter"
        batch = kwargs["texts"]
        out = []
        for idx, _text in enumerate(batch):
            out.append([float(idx + 1)] * 768)
        return np.asarray(out, dtype=np.float32), {"transport": "hf_endpoint", "texts_total": int(len(batch))}

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(precompute_module, "embed_texts_via_hf_endpoint", _fake_embed_texts_via_hf_endpoint)

    output_root = tmp_path / "precomputed"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "precompute-source-embeddings",
            "--publications-path",
            str(publications_path),
            "--output-root",
            str(output_root),
            "--no-progress",
        ]
    )
    payload = args.func(args)

    enriched = pd.read_parquet(output_root / "publications_precomputed.parquet")
    report = json.loads((output_root / "precompute_source_embeddings_report.json").read_text(encoding="utf-8"))

    assert payload["publications_output_path"] == str(output_root / "publications_precomputed.parquet")
    assert len(enriched) == 2
    assert "precomputed_embedding" in enriched.columns
    assert len(enriched.loc[0, "precomputed_embedding"]) == 768
    np.testing.assert_allclose(np.asarray(enriched.loc[1, "precomputed_embedding"], dtype=np.float32), np.array([0.25] * 768))
    assert report["datasets"]["publications"]["reused_precomputed_count"] == 1
    assert report["datasets"]["publications"]["missing_precomputed_count"] == 1
    assert report["runtime"]["texts_remote_computed"] == 1

    bundle_dir = _write_bundle(tmp_path)
    _apply_fast_infer_mocks(monkeypatch)

    infer_output_root = tmp_path / "infer_out"
    result = run_infer_sources(
        InferSourcesRequest(
            publications_path=output_root / "publications_precomputed.parquet",
            output_root=infer_output_root,
            dataset_id="my_ads_2026",
            model_bundle=bundle_dir,
            device="cpu",
            cluster_backend="sklearn_cpu",
            progress=False,
        )
    )
    stage_metrics = json.loads(result.stage_metrics_path.read_text(encoding="utf-8"))

    assert stage_metrics["precomputed_embeddings"]["mentions"]["used_precomputed_embeddings"] is True
    assert stage_metrics["runtime"]["specter"]["generation_mode"] == "precomputed_only"
