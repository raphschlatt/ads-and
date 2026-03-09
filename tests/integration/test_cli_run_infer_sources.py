from __future__ import annotations

import json
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


def _apply_fast_mocks(monkeypatch) -> None:
    def _chars(mentions, output_path, force_recompute=False, **_kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force_recompute:
            arr = np.load(path)
            meta = {
                "cache_hit": True,
                "generation_mode": "cache",
                "name_count": int(len(mentions)),
            }
            return (arr, meta) if _kwargs.get("return_meta") else arr
        arr = np.ones((len(mentions), 50), dtype=np.float32)
        np.save(path, arr)
        meta = {
            "cache_hit": False,
            "generation_mode": "chars2vec",
            "name_count": int(len(mentions)),
        }
        return (arr, meta) if _kwargs.get("return_meta") else arr

    def _text(mentions, output_path, force_recompute=False, **_kwargs):
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
            "effective_precision_mode": None,
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
        }
        return (out, meta) if _kwargs.get("return_runtime_meta") else out

    def _cluster(mentions, pair_scores, output_path=None, **_kwargs):
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
            return out, {"cluster_backend_effective": "sklearn_cpu", "cpu_workers_effective": 1}
        return out

    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_specter_embeddings", _text)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_with_checkpoint", _score)
    monkeypatch.setattr("author_name_disambiguation.source_inference.cluster_blockwise_dbscan", _cluster)


def test_cli_run_infer_sources_writes_artifacts(monkeypatch, tmp_path: Path, capsys):
    publications_path, references_path = _write_dataset(tmp_path, with_references=True)
    bundle_dir = _write_bundle(tmp_path)
    _apply_fast_mocks(monkeypatch)

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
    assert context["runtime"]["specter"]["fallback_reason"] == "torch_cuda_unavailable"
    assert preflight["runtime"]["pair_scoring"]["resolved_device"] == "cpu"
    assert stage_metrics["runtime"]["specter"]["requested_device"] == "auto"
    assert stage_metrics["precomputed_embeddings"]["mentions"]["precomputed_embedding_count"] == 0
    assert "START Bootstrap and context" in captured.err
    assert "START SPECTER embeddings" in captured.err
    assert "START Export and reports" in captured.err
    assert "\r" not in captured.err
    payload_stdout = json.loads(captured.out)
    assert payload_stdout["run_id"] == payload["run_id"]


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
