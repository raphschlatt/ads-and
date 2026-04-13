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
    def _policy(
        *,
        requested_device,
        runtime_mode_requested,
        specter_runtime_backend_requested,
        cluster_backend_requested,
        score_batch_size,
        scratch_dir,
        bootstrap_runtime=None,
    ):
        bootstrap = dict(bootstrap_runtime or {})
        runtime_mode_effective = (
            str(runtime_mode_requested).strip().lower()
            if runtime_mode_requested is not None
            else ("gpu" if str(bootstrap.get("resolved_device") or "").startswith("cuda") else "cpu")
        )
        if runtime_mode_effective == "cpu":
            specter_backend = str(specter_runtime_backend_requested or "cpu_auto")
            effective_device = "cpu"
        else:
            specter_backend = "transformers"
            effective_device = "cuda"
        cluster_requested = str(cluster_backend_requested or "sklearn_cpu")
        cluster_effective = "sklearn_cpu" if cluster_requested == "auto" else cluster_requested
        safety_fallbacks = []
        if runtime_mode_effective == "cpu" and str(requested_device) == "auto":
            safety_fallbacks.append(
                {
                    "component": "runtime_mode",
                    "reason": str(bootstrap.get("fallback_reason") or "torch_cuda_unavailable"),
                    "action": "cpu_runtime_selected",
                }
            )
        return {
            "host_profile": {
                "requested_device": str(requested_device),
                "available_ram_bytes": 16 * 1024**3,
                "scratch_dir": str(scratch_dir),
                "scratch_free_bytes": 100 * 1024**3,
                "cpu": {"cpu_limit": 4, "cpu_limit_source": "test"},
                "torch": {
                    "torch_version": bootstrap.get("torch_version"),
                    "torch_cuda_version": bootstrap.get("torch_cuda_version"),
                    "torch_cuda_available": bootstrap.get("torch_cuda_available"),
                    "resolved_device": bootstrap.get("resolved_device"),
                    "gpu_name": bootstrap.get("gpu_name"),
                    "gpu_total_memory_bytes": None,
                    "fallback_reason": bootstrap.get("fallback_reason"),
                    "cuda_probe_error": bootstrap.get("cuda_probe_error"),
                },
                "tensorflow_runtime": {"status": "cpu_fallback", "reason": "forced_cpu"},
                "onnx_cpu_backend": {"available": True, "reason": None},
                "cuml_gpu_backend": {"available": False, "reason": "missing"},
            },
            "resolved_runtime_policy": {
                "runtime_mode_requested": runtime_mode_requested,
                "runtime_mode_effective": runtime_mode_effective,
                "requested_device": str(requested_device),
                "effective_request_device": effective_device,
                "specter_runtime_backend_requested": specter_runtime_backend_requested,
                "specter_runtime_backend_effective": specter_backend,
                "chars2vec_force_cpu": True,
                "chars2vec_batch_size": 128,
                "score_batch_size_requested": int(score_batch_size),
                "score_batch_size_effective": int(score_batch_size),
                "cluster_backend_requested": cluster_requested,
                "cluster_backend_effective": cluster_effective,
                "exact_graph_union_impl": "python",
                "numba_auto_enabled": False,
                "onnx_cpu_available": True,
                "cuml_gpu_available": False,
            },
            "safety_fallbacks": safety_fallbacks,
        }

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

    def _encode_mentions(chars2vec, source_text_embeddings, mention_source_index, output_path, norms_output_path=None, **_kwargs):
        del chars2vec, source_text_embeddings
        mention_index = np.load(mention_source_index)
        arr = np.ones((len(mention_index), 4), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1).astype(np.float32)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        norms_path = Path(norms_output_path) if norms_output_path is not None else Path(output_path).with_name("mention_embeddings_norms.npy")
        np.save(norms_path, norms)
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
            "mention_encode_seconds": 0.01,
            "mention_embedding_shape": [int(len(arr)), int(arr.shape[1])],
            "mention_embedding_bytes": int(arr.nbytes),
            "mention_norm_bytes": int(norms.nbytes),
            "cuda_oom_fallback_used": False,
        }
        return (Path(output_path), meta) if _kwargs.get("return_runtime_meta") else Path(output_path)

    def _score_from_embeddings(mentions, pairs, output_path=None, score_callback=None, **_kwargs):
        out, meta = _score(mentions=mentions, pairs=pairs, output_path=output_path, **_kwargs)
        if score_callback is not None:
            score_callback(
                {
                    "pair_id": out["pair_id"].astype(str).to_numpy(copy=False),
                    "mention_id_1": out["mention_id_1"].astype(str).to_numpy(copy=False),
                    "mention_id_2": out["mention_id_2"].astype(str).to_numpy(copy=False),
                    "block_key": out["block_key"].astype(str).to_numpy(copy=False),
                    "distance": out["distance"].astype(np.float32).to_numpy(copy=False),
                }
            )
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

    class _FakeExactGraphClusterAccumulator:
        def __init__(self, *, mentions, cluster_config, backend_requested="auto"):
            del cluster_config
            self._mentions = mentions
            self._backend_requested = backend_requested
            self._pair_rows = 0

        def consume_score_columns(self, score_columns):
            self._pair_rows += int(len(score_columns.get("block_key", [])))

        def finalize(self):
            out = self._mentions[["mention_id", "block_key"]].copy()
            out["author_uid"] = [f"uid.{idx}" for idx in range(len(out))]
            meta = {
                "cluster_backend_requested": str(self._backend_requested),
                "cluster_backend_effective": "connected_components_cpu",
                "cluster_backend_reason": "test",
                "cpu_sharding_mode": "off",
                "cpu_sharding_enabled": False,
                "cpu_workers_requested": "auto",
                "cpu_workers_effective": 2,
                "ram_budget_bytes": None,
                "total_pairs_est": int(self._pair_rows),
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
                "sanitize_totals": {
                    "corrected_blocks": 0,
                    "non_finite_count": 0,
                    "negative_count": 0,
                    "above_max_count": 0,
                    "asymmetry_pairs": 0,
                    "diag_reset_count": 0,
                },
            }
            return out, meta

    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_specter_embeddings", text_mock)
    monkeypatch.setattr("author_name_disambiguation.source_inference.build_pairs_within_blocks", _pairs)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_with_checkpoint", _score)
    monkeypatch.setattr("author_name_disambiguation.source_inference.cluster_blockwise_dbscan", _cluster)
    monkeypatch.setattr("author_name_disambiguation.source_inference.encode_mentions_to_memmap", _encode_mentions)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_from_mention_embeddings", _score_from_embeddings)
    monkeypatch.setattr("author_name_disambiguation.source_inference.ExactGraphClusterAccumulator", _FakeExactGraphClusterAccumulator)
    monkeypatch.setattr("author_name_disambiguation.source_inference.resolve_infer_runtime_policy", _policy)
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
    assert stage_metrics["resolved_runtime_policy"]["runtime_mode_effective"] == "cpu"
    assert stage_metrics["resolved_runtime_policy"]["specter_runtime_backend_effective"] == "cpu_auto"
    assert any(item["component"] == "specter" for item in stage_metrics["safety_fallbacks"])
    assert stage_metrics["runtime"]["specter"]["runtime_mode"] == "cpu"
    assert stage_metrics["runtime"]["specter"]["runtime_backend"] == "transformers"
    assert stage_metrics["runtime"]["specter"]["fallback_reason"] == "cpu_auto_onnx_fallback"


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
    assert context["resolved_runtime_policy"]["runtime_mode_effective"] == "cpu"
    assert context["host_profile"]["cpu"]["cpu_limit"] == 4
    assert "legacy_runtime_overrides" not in context
    assert "legacy_runtime_overrides" not in stage_metrics["runtime"]["specter"]
