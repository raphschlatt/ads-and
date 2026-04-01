from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from author_name_disambiguation import cli
from author_name_disambiguation import source_inference
from author_name_disambiguation.infer_sources import InferSourcesRequest, run_infer_sources


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def _write_dataset(tmp_path: Path, *, with_references: bool = True) -> tuple[Path, Path | None]:
    pubs_path = tmp_path / "publications.parquet"
    refs_path = tmp_path / "references.parquet"

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
                "Author": ["Doe J."],
                "Title_en": "Paper 2",
                "Abstract_en": "Abstract 2",
                "Year": 2021,
                "Affiliation": ["Inst C"],
            },
        ]
    ).to_parquet(pubs_path, index=False)

    if with_references:
        pd.DataFrame(
            [
                {
                    "Bibcode": "bib3",
                    "Author": ["Ref X", "Ref Y"],
                    "Title_en": "Paper 3",
                    "Abstract_en": "Abstract 3",
                    "Year": 2022,
                    "Affiliation": ["Inst R1", "Inst R2"],
                }
            ]
        ).to_parquet(refs_path, index=False)
        return pubs_path, refs_path

    return pubs_path, None


def _write_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "checkpoint.pt").write_text("checkpoint", encoding="utf-8")
    _write_yaml(
        bundle_dir / "model_config.yaml",
        {
            "name": "mock-nand",
            "representation": {"text_model_name": "mock-specter", "max_length": 64},
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


def _apply_fast_mocks(monkeypatch, *, empty_chunked_score_return: bool = False) -> dict[str, object]:
    seen: dict[str, object] = {}

    def _chars(mentions, output_path, force_recompute=False, **_kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force_recompute:
            arr = np.load(path)
            meta = {
                "cache_hit": True,
                "generation_mode": "cache",
                "name_count": int(len(mentions)),
                "unique_name_count": int(len(mentions)),
                "wall_seconds": 0.01,
                "generation_seconds": 0.0,
                "model_load_seconds": 0.0,
                "normalize_seconds": 0.0,
                "unique_seconds": 0.0,
                "pad_seconds": 0.0,
                "predict_seconds": 0.0,
                "materialize_seconds": 0.0,
                "cache_load_seconds": 0.01,
                "cache_write_seconds": 0.0,
                "tensorflow_memory_growth_enabled": None,
                "tensorflow_memory_growth_error": None,
                "tensorflow_cleanup_attempted": False,
                "tensorflow_cleanup_error": None,
            }
            return (arr, meta) if _kwargs.get("return_meta") else arr
        arr = np.ones((len(mentions), 50), dtype=np.float32)
        np.save(path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "chars2vec",
            "name_count": int(len(mentions)),
            "unique_name_count": int(len(mentions)),
            "wall_seconds": 0.02,
            "generation_seconds": 0.01,
            "model_load_seconds": 0.001,
            "normalize_seconds": 0.001,
            "unique_seconds": 0.001,
            "pad_seconds": 0.003,
            "predict_seconds": 0.004,
            "materialize_seconds": 0.001,
            "cache_load_seconds": 0.0,
            "cache_write_seconds": 0.01,
            "tensorflow_memory_growth_enabled": True,
            "tensorflow_memory_growth_error": None,
            "tensorflow_cleanup_attempted": True,
            "tensorflow_cleanup_error": None,
        }
        return (arr, meta) if _kwargs.get("return_meta") else arr

    def _text(mentions, output_path, force_recompute=False, **_kwargs):
        seen["specter_kwargs"] = dict(_kwargs)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force_recompute:
            arr = np.load(path)
            meta = {
                "cache_hit": True,
                "generation_mode": "cache",
                "requested_device": str(_kwargs.get("device", "auto")),
                "resolved_device": None,
                "fallback_reason": None,
                "torch_version": None,
                "torch_cuda_version": None,
                "torch_cuda_available": None,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": None,
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
                "column_present": False,
                "precomputed_embedding_count": 0,
                "recomputed_embedding_count": len(mentions),
                "used_precomputed_embeddings": False,
            }
            return (arr, meta) if _kwargs.get("return_meta") else arr
        arr = np.ones((len(mentions), 768), dtype=np.float32)
        np.save(path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "model_only",
            "requested_device": str(_kwargs.get("device", "auto")),
            "resolved_device": "cpu",
            "fallback_reason": "torch_cuda_unavailable",
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "cuda_probe_error": None,
            "model_to_cuda_error": None,
            "effective_precision_mode": str(_kwargs.get("precision_mode", "auto")),
            "requested_batch_size": _kwargs.get("batch_size"),
            "effective_batch_size": _kwargs.get("batch_size"),
            "oom_retry_count": 0,
            "batches_total": max(0, len(mentions)),
            "tokenize_seconds_total": 0.0,
            "host_to_device_seconds_total": 0.0,
            "forward_seconds_total": 0.0,
            "device_to_host_seconds_total": 0.0,
            "token_count_total": int(len(mentions)),
            "max_sequence_length_observed": 1,
            "mean_sequence_length_observed": 1.0,
            "device_to_host_flushes": max(0, len(mentions)),
            "column_present": False,
            "precomputed_embedding_count": 0,
            "recomputed_embedding_count": len(mentions),
            "used_precomputed_embeddings": False,
        }
        return (arr, meta) if _kwargs.get("return_meta") else arr

    def _score(mentions, pairs, output_path=None, **_kwargs):
        if isinstance(pairs, (str, Path)):
            pairs_df = pd.read_parquet(pairs)
        else:
            pairs_df = pairs
        out = pairs_df[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
        out["cosine_sim"] = pd.Series(np.linspace(0.7, 0.9, num=len(out), dtype=np.float32))
        out["distance"] = 1.0 - out["cosine_sim"]
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        meta = {
            "requested_device": str(_kwargs.get("device", "auto")),
            "resolved_device": "cpu",
            "fallback_reason": "torch_cuda_unavailable",
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "cuda_probe_error": None,
            "model_to_cuda_error": None,
            "effective_precision_mode": str(_kwargs.get("precision_mode", "fp32")),
            "pair_scoring_strategy": "preencoded_mentions",
            "mention_storage_device": "cpu",
            "cuda_oom_fallback_used": False,
            "numeric_clamping": {
                "clamped": False,
                "events": 0,
                "cosine_non_finite_count": 0,
                "cosine_below_min_count": 0,
                "cosine_above_max_count": 0,
                "distance_non_finite_count": 0,
                "distance_below_min_count": 0,
                "distance_above_max_count": 0,
            },
        }
        if empty_chunked_score_return and isinstance(pairs, (str, Path)) and not _kwargs.get("return_scores", True):
            empty = pd.DataFrame(columns=out.columns)
            return (empty, meta) if _kwargs.get("return_runtime_meta") else empty
        return (out, meta) if _kwargs.get("return_runtime_meta") else out

    def _pairs(mentions, output_path=None, **_kwargs):
        seen["pair_kwargs"] = dict(_kwargs)
        rows = []
        for block_key, grp in mentions.groupby("block_key", sort=False):
            if len(grp) < 2:
                continue
            base = grp.iloc[0]
            other = grp.iloc[1]
            rows.append(
                {
                    "pair_id": f"{base['mention_id']}__{other['mention_id']}",
                    "mention_id_1": str(base["mention_id"]),
                    "mention_id_2": str(other["mention_id"]),
                    "block_key": str(block_key),
                    "split": str(base.get("split", "inference")),
                    "label": None,
                }
            )
        out = pd.DataFrame(rows)
        meta = {
            "same_publication_pairs_skipped": 0,
            "pairs_written": int(len(out)),
            "chunk_rows": int(_kwargs.get("chunk_rows", 0)),
            "output_path": None if output_path is None else str(output_path),
            "cpu_sharding_mode": str(_kwargs.get("sharding_mode", "auto")),
            "cpu_sharding_enabled": str(_kwargs.get("sharding_mode", "auto")) != "off",
            "cpu_workers_requested": "auto" if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
            "cpu_workers_effective": 4 if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
            "cpu_limit_detected": 8,
            "cpu_limit_source": "test",
            "cpu_min_pairs_per_worker": int(_kwargs.get("min_pairs_per_worker", 1)),
            "ram_budget_bytes": int(_kwargs["ram_budget_bytes"]) if _kwargs.get("ram_budget_bytes") is not None else None,
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
            "block_size_histogram": {"2": 1, "3": 1},
        }
        return (out, meta) if _kwargs.get("return_meta") else out

    def _cluster(mentions, pair_scores, output_path=None, **_kwargs):
        seen["cluster_kwargs"] = dict(_kwargs)
        seen["cluster_pair_scores_rows"] = int(len(pair_scores))
        out = mentions[["mention_id", "block_key"]].copy()
        out["author_uid"] = [
            "blk.a.0",
            "blk.a.1",
            "blk.a.0",
            "blk.r.0",
            "blk.r.1",
        ][: len(out)]
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        if bool(_kwargs.get("return_meta")):
            return out, {
                "build_entries_seconds": 0.01,
                "cluster_backend_requested": str(_kwargs.get("backend", "auto")),
                "cluster_backend_effective": "sklearn_cpu",
                "cluster_backend_reason": "test",
                "cpu_sharding_mode": str(_kwargs.get("sharding_mode", "auto")),
                "cpu_sharding_enabled": str(_kwargs.get("sharding_mode", "auto")) != "off",
                "cpu_workers_requested": "auto" if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
                "cpu_workers_effective": 4 if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
                "ram_budget_bytes": int(_kwargs["ram_budget_bytes"]) if _kwargs.get("ram_budget_bytes") is not None else None,
                "total_pairs_est": int(len(pair_scores)),
                "block_p95": 2.0,
                "block_size_histogram": {"1": 1, "2": 2},
                "block_count_by_bucket": {"1": 1, "2": 2},
                "total_seconds_by_bucket": {"1": 0.002, "2": 0.008},
                "dbscan_seconds_by_bucket": {"1": 0.0, "2": 0.01},
                "distance_matrix_seconds_total": 0.01,
                "constraints_seconds_total": 0.0,
                "sanitize_seconds_total": 0.0,
                "dbscan_seconds_total": 0.01,
                "gpu_transfer_seconds_total": 0.0,
                "top_slow_blocks": [],
            }
        return out

    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_specter_embeddings", _text)
    monkeypatch.setattr("author_name_disambiguation.source_inference.build_pairs_within_blocks", _pairs)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_with_checkpoint", _score)
    monkeypatch.setattr("author_name_disambiguation.source_inference.cluster_blockwise_dbscan", _cluster)
    return seen


def test_cli_run_infer_sources_writes_artifacts(monkeypatch, tmp_path: Path, capsys):
    publications_path, references_path = _write_dataset(tmp_path, with_references=True)
    bundle_dir = _write_bundle(tmp_path)
    seen = _apply_fast_mocks(monkeypatch)
    monkeypatch.setattr(
        "author_name_disambiguation.source_inference._probe_bootstrap_runtime",
        lambda _device: {
            "requested_device": "auto",
            "resolved_device": "cuda:0",
            "gpu_name": "NVIDIA A100 80GB PCIe",
            "torch_version": "2.10.0+cu126",
            "torch_cuda_version": "12.6",
            "torch_cuda_available": True,
            "fallback_reason": None,
            "cuda_probe_error": None,
        },
    )

    output_root = tmp_path / "out"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            str(publications_path),
            "--references-path",
            str(references_path),
            "--output-root",
            str(output_root),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(bundle_dir),
        ]
    )
    payload = args.func(args)
    captured = capsys.readouterr()

    assert payload["run_id"]
    assert (output_root / "mention_clusters.parquet").exists()
    assert (output_root / "source_author_assignments.parquet").exists()
    assert (output_root / "author_entities.parquet").exists()
    assert (output_root / "05_stage_metrics_infer_sources.json").exists()
    assert (output_root / "05_go_no_go_infer_sources.json").exists()
    assert (output_root / "publications_disambiguated.parquet").exists()
    assert (output_root / "references_disambiguated.parquet").exists()
    context = json.loads((output_root / "00_context.json").read_text(encoding="utf-8"))
    preflight = json.loads((output_root / "02_preflight_infer.json").read_text(encoding="utf-8"))
    stage_metrics = json.loads((output_root / "05_stage_metrics_infer_sources.json").read_text(encoding="utf-8"))

    pubs_df = pd.read_parquet(output_root / "publications_disambiguated.parquet")
    refs_df = pd.read_parquet(output_root / "references_disambiguated.parquet")
    entities = pd.read_parquet(output_root / "author_entities.parquet")
    assignments = pd.read_parquet(output_root / "source_author_assignments.parquet")

    assert "AuthorUID" in pubs_df.columns
    assert "AuthorDisplayName" in pubs_df.columns
    assert "AuthorUID" in refs_df.columns
    assert pubs_df.loc[0, "AuthorUID"][0].startswith("my_ads_2026::")
    assert refs_df.loc[0, "AuthorUID"][1].startswith("my_ads_2026::blk.r.1")
    assert "author_display_name" in entities.columns
    assert set(assignments["assignment_kind"].unique()) == {"canonical"}
    assert context["runtime"]["load_inputs"]["read_publications_seconds"] >= 0.0
    assert context["runtime"]["chars2vec"]["generation_mode"] == "chars2vec"
    assert context["runtime"]["chars2vec"]["wall_seconds"] >= 0.0
    assert context["runtime"]["chars2vec"]["unique_name_count"] == 5
    assert context["runtime"]["specter"]["fallback_reason"] == "torch_cuda_unavailable"
    assert context["runtime"]["pair_building"]["cpu_workers_effective"] == 4
    assert context["runtime"]["clustering"]["cpu_workers_effective"] == 4
    assert preflight["runtime"]["chars2vec"]["tensorflow_cleanup_attempted"] is True
    assert preflight["runtime"]["load_inputs"]["explode_mentions_seconds"] >= 0.0
    assert preflight["runtime"]["pair_scoring"]["resolved_device"] == "cpu"
    assert preflight["runtime"]["pair_building"]["cpu_sharding_enabled"] is True
    assert preflight["runtime"]["clustering"]["cpu_sharding_enabled"] is True
    assert preflight["runtime"]["export"]["mirror_mode"] == "parquet_frame_reuse"
    assert stage_metrics["runtime"]["chars2vec"]["generation_mode"] == "chars2vec"
    assert stage_metrics["runtime"]["load_inputs"]["deduplicate_seconds"] >= 0.0
    assert stage_metrics["runtime"]["specter"]["runtime_mode"] == "gpu"
    assert stage_metrics["runtime"]["specter"]["runtime_backend"] == "transformers"
    assert stage_metrics["runtime"]["specter"]["resolved_device"] == "cpu"
    assert "device_to_host_flushes" in stage_metrics["runtime"]["specter"]
    assert "token_count_total" in stage_metrics["runtime"]["specter"]
    assert stage_metrics["runtime"]["pair_building"]["cpu_workers_requested"] == "auto"
    assert "group_blocks_seconds" in stage_metrics["runtime"]["pair_building"]
    assert "top_slow_blocks" in stage_metrics["runtime"]["pair_building"]
    assert stage_metrics["runtime"]["clustering"]["block_size_histogram"]
    assert stage_metrics["runtime"]["clustering"]["block_count_by_bucket"]
    assert stage_metrics["runtime"]["clustering"]["total_seconds_by_bucket"]
    assert stage_metrics["runtime"]["clustering"]["dbscan_seconds_by_bucket"]
    assert stage_metrics["runtime"]["export"]["source_reread_seconds"] == 0.0
    assert stage_metrics["precomputed_embeddings"]["mentions"]["precomputed_embedding_count"] == 0
    assert "START Bootstrap" in captured.err
    assert "INFO device=auto -> cuda:0 | gpu=NVIDIA A100 80GB PCIe | precision=fp32 | torch=2.10.0+cu126" in captured.err
    assert "START Load inputs" in captured.err
    assert "START Preflight" in captured.err
    assert "START Name embeddings" in captured.err
    assert "DONE 5 names embedded in " in captured.err
    assert "START Text embeddings" in captured.err
    assert "DONE 5 texts embedded in " in captured.err
    assert "cpu_stage=pair_building | cpu_workers=auto | sharding=auto | ram_budget=" in captured.err
    assert "pair_count=" in captured.err
    assert "pairs_est=" in captured.err
    assert "workers=4 | sharding=yes | score_count=" in captured.err
    assert "backend=auto | cpu_workers=auto | sharding=auto | ram_budget=" in captured.err
    assert "backend=sklearn_cpu | workers=4 | sharding=yes" in captured.err
    assert "START Export and reports" in captured.err
    assert "Run complete | run_id=" in captured.err
    assert "\r" not in captured.err
    payload_stdout = json.loads(captured.out)
    assert payload_stdout["run_id"] == payload["run_id"]
    assert seen["specter_kwargs"]["reuse_model"] is False
    assert seen["pair_kwargs"]["num_workers"] is None
    assert seen["pair_kwargs"]["sharding_mode"] == "auto"
    assert seen["pair_kwargs"]["min_pairs_per_worker"] == 1_000_000
    assert int(seen["pair_kwargs"]["ram_budget_bytes"]) > 0
    assert seen["cluster_kwargs"]["num_workers"] is None
    assert seen["cluster_kwargs"]["sharding_mode"] == "auto"
    assert seen["cluster_kwargs"]["min_pairs_per_worker"] == 1_000_000
    assert int(seen["cluster_kwargs"]["ram_budget_bytes"]) > 0


def test_cli_run_infer_sources_passes_specter_overrides(monkeypatch, tmp_path: Path):
    publications_path, references_path = _write_dataset(tmp_path, with_references=True)
    bundle_dir = _write_bundle(tmp_path)
    seen = _apply_fast_mocks(monkeypatch)
    monkeypatch.setattr(
        "author_name_disambiguation.source_inference._probe_bootstrap_runtime",
        lambda _device: {
            "requested_device": "auto",
            "resolved_device": "cpu",
            "gpu_name": None,
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "fallback_reason": "torch_cuda_unavailable",
            "cuda_probe_error": None,
        },
    )
    original_resolve_infer_run_cfg = source_inference._resolve_infer_run_cfg

    def _patched_resolve_infer_run_cfg(infer_stage):
        cfg = original_resolve_infer_run_cfg(infer_stage)
        cfg["infer_overrides"] = dict(cfg.get("infer_overrides", {}) or {})
        cfg["infer_overrides"]["specter_batch_size"] = 48
        cfg["infer_overrides"]["specter_precision_mode"] = "amp_bf16"
        return cfg

    monkeypatch.setattr(
        "author_name_disambiguation.source_inference._resolve_infer_run_cfg",
        _patched_resolve_infer_run_cfg,
    )

    output_root = tmp_path / "out_specter_overrides"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            str(publications_path),
            "--references-path",
            str(references_path),
            "--output-root",
            str(output_root),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(bundle_dir),
            "--no-progress",
        ]
    )
    args.func(args)

    preflight = json.loads((output_root / "02_preflight_infer.json").read_text(encoding="utf-8"))
    assert seen["specter_kwargs"]["batch_size"] == 48
    assert seen["specter_kwargs"]["precision_mode"] == "amp_bf16"
    assert seen["specter_kwargs"]["reuse_model"] is False
    assert preflight["runtime"]["specter"]["effective_precision_mode"] == "amp_bf16"
    assert preflight["runtime"]["specter"]["requested_batch_size"] == 48


def test_cli_and_api_infer_sources_parity(monkeypatch, tmp_path: Path):
    publications_path, references_path = _write_dataset(tmp_path, with_references=False)
    bundle_dir = _write_bundle(tmp_path)
    _apply_fast_mocks(monkeypatch)

    output_root_cli = tmp_path / "out_cli"
    output_root_api = tmp_path / "out_api"

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            str(publications_path),
            "--output-root",
            str(output_root_cli),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(bundle_dir),
            "--no-progress",
        ]
    )
    args.func(args)

    api_result = run_infer_sources(
        InferSourcesRequest(
            publications_path=publications_path,
            output_root=output_root_api,
            dataset_id="my_ads_2026",
            model_bundle=bundle_dir,
            progress=False,
        )
    )

    cli_stage = json.loads((output_root_cli / "05_stage_metrics_infer_sources.json").read_text(encoding="utf-8"))
    api_stage = json.loads((output_root_api / "05_stage_metrics_infer_sources.json").read_text(encoding="utf-8"))

    assert api_result.stage_metrics_path == output_root_api / "05_stage_metrics_infer_sources.json"
    assert cli_stage["metric_scope"] == api_stage["metric_scope"] == "infer"
    assert cli_stage["infer_stage"] == api_stage["infer_stage"] == "full"
    assert cli_stage["counts"]["ads_mentions"] == api_stage["counts"]["ads_mentions"]


def test_cli_run_infer_sources_reloads_file_backed_pair_scores(monkeypatch, tmp_path: Path):
    publications_path, references_path = _write_dataset(tmp_path, with_references=True)
    bundle_dir = _write_bundle(tmp_path)
    seen = _apply_fast_mocks(monkeypatch, empty_chunked_score_return=True)
    monkeypatch.setattr(
        "author_name_disambiguation.source_inference._probe_bootstrap_runtime",
        lambda _device: {
            "requested_device": "auto",
            "resolved_device": "cpu",
            "gpu_name": None,
            "torch_version": "2.10.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "fallback_reason": "torch_cuda_unavailable",
            "cuda_probe_error": None,
        },
    )
    original_resolve_infer_run_cfg = source_inference._resolve_infer_run_cfg

    def _patched_resolve_infer_run_cfg(infer_stage):
        cfg = original_resolve_infer_run_cfg(infer_stage)
        cfg["infer_overrides"] = dict(cfg.get("infer_overrides", {}) or {})
        cfg["infer_overrides"]["score_chunked_threshold"] = 1
        return cfg

    monkeypatch.setattr(
        "author_name_disambiguation.source_inference._resolve_infer_run_cfg",
        _patched_resolve_infer_run_cfg,
    )

    output_root = tmp_path / "out_chunked"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            str(publications_path),
            "--references-path",
            str(references_path),
            "--output-root",
            str(output_root),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(bundle_dir),
            "--no-progress",
        ]
    )
    args.func(args)

    preflight = json.loads((output_root / "02_preflight_infer.json").read_text(encoding="utf-8"))
    assert seen["cluster_pair_scores_rows"] > 0
    assert int(preflight["runtime"]["pair_building"]["pairs_written"]) == int(seen["cluster_pair_scores_rows"])


def test_cli_run_infer_sources_emits_single_aggregated_pair_clamp_warning(monkeypatch, tmp_path: Path, capsys):
    publications_path, references_path = _write_dataset(tmp_path, with_references=True)
    bundle_dir = _write_bundle(tmp_path)
    _apply_fast_mocks(monkeypatch)

    def _score_with_clamp_meta(mentions, pairs, output_path=None, **_kwargs):
        if isinstance(pairs, (str, Path)):
            pairs_df = pd.read_parquet(pairs)
        else:
            pairs_df = pairs
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
            "numeric_clamping": {
                "clamped": True,
                "events": 3,
                "cosine_non_finite_count": 0,
                "cosine_below_min_count": 0,
                "cosine_above_max_count": 11,
                "distance_non_finite_count": 0,
                "distance_below_min_count": 0,
                "distance_above_max_count": 0,
            },
        }
        return (out, meta) if _kwargs.get("return_runtime_meta") else out

    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_with_checkpoint", _score_with_clamp_meta)

    output_root = tmp_path / "out_warn"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            str(publications_path),
            "--references-path",
            str(references_path),
            "--output-root",
            str(output_root),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(bundle_dir),
        ]
    )
    args.func(args)
    captured = capsys.readouterr()

    assert captured.err.count("WARN Applied numeric clamping to pair scores:") == 1
    assert "RuntimeWarning: Applied numeric clamping" not in captured.err


def test_cli_compare_infer_baseline_writes_deep_compare_report(tmp_path: Path, capsys):
    metrics_root = tmp_path / "exports"
    baseline_dir = metrics_root / "baseline_run"
    current_dir = metrics_root / "current_run"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)

    with (baseline_dir / "05_stage_metrics_infer_sources.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric_scope": "infer",
                "run_id": "baseline_internal",
                "counts": {"ads_mentions": 3, "ads_clusters": 2, "ads_cluster_assignments": 3},
                "singleton_ratio": 0.2,
                "split_high_sim_rate_probe": 0.1,
                "merged_low_conf_rate_probe": 0.2,
                "source_export": {"coverage_rate": 1.0},
            },
            handle,
            indent=2,
        )
    with (current_dir / "05_stage_metrics_infer_sources.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric_scope": "infer",
                "run_id": "current_internal",
                "counts": {"ads_mentions": 3, "ads_clusters": 1, "ads_cluster_assignments": 3},
                "singleton_ratio": 0.1,
                "split_high_sim_rate_probe": 0.05,
                "merged_low_conf_rate_probe": 0.25,
                "source_export": {"coverage_rate": 1.0},
            },
            handle,
            indent=2,
        )
    with (baseline_dir / "05_go_no_go_infer_sources.json").open("w", encoding="utf-8") as handle:
        json.dump({"go": True, "warnings": []}, handle, indent=2)
    with (current_dir / "05_go_no_go_infer_sources.json").open("w", encoding="utf-8") as handle:
        json.dump({"go": True, "warnings": []}, handle, indent=2)
    pd.DataFrame(
        [
            {"mention_id": "m1", "block_key": "blk.a", "author_uid_local": "blk.a::0"},
            {"mention_id": "m2", "block_key": "blk.a", "author_uid_local": "blk.a::0"},
            {"mention_id": "m3", "block_key": "blk.a", "author_uid_local": "blk.a::1"},
        ]
    ).to_parquet(baseline_dir / "mention_clusters.parquet", index=False)
    pd.DataFrame(
        [
            {"mention_id": "m1", "block_key": "blk.a", "author_uid_local": "blk.a::0"},
            {"mention_id": "m2", "block_key": "blk.a", "author_uid_local": "blk.a::0"},
            {"mention_id": "m3", "block_key": "blk.a", "author_uid_local": "blk.a::0"},
        ]
    ).to_parquet(current_dir / "mention_clusters.parquet", index=False)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "compare-infer-baseline",
            "--baseline-run-id",
            "baseline_run",
            "--current-run-id",
            "current_run",
            "--metrics-root",
            str(metrics_root),
            "--no-progress",
        ]
    )
    payload = args.func(args)
    captured = capsys.readouterr()

    assert payload["mention_cluster_changed_mentions"] == 1
    assert payload["mention_cluster_changed_blocks"] == 1
    assert (current_dir / "99_compare_infer_to_baseline.json").exists()
    payload_stdout = json.loads(captured.out)
    assert payload_stdout["output_path"].endswith("99_compare_infer_to_baseline.json")
