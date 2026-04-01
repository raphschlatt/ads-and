from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from author_name_disambiguation.common.pipeline_reports import default_run_id, write_json
from author_name_disambiguation.data.prepare_ads import load_ads_records
from author_name_disambiguation.embedding_contract import build_bundle_embedding_contract
from author_name_disambiguation.features.embed_specter import generate_specter_embeddings
from author_name_disambiguation.hf_transport import embed_texts_via_hf_httpx, normalize_model_name, normalize_provider
from author_name_disambiguation.infer_sources import InferSourcesRequest, run_infer_sources
from author_name_disambiguation.source_inference import _resolve_model_bundle


@dataclass(slots=True)
class HfCompatibilityReportRequest:
    publications_path: str | Path
    output_root: str | Path
    dataset_id: str
    model_bundle: str | Path
    references_path: str | Path | None = None
    sample_size: int = 128
    provider: str = "hf-inference"
    model_name: str = "allenai/specter"
    hf_token_env_var: str = "HF_TOKEN"
    batch_size: int = 32
    device: str = "auto"
    force: bool = False
    progress: bool = True


@dataclass(slots=True)
class HfCompatibilityReportResult:
    run_id: str
    output_root: Path
    report_json_path: Path
    report_markdown_path: Path
    compatible: bool


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _load_normalized_source(path: str | Path, *, source_type: str) -> pd.DataFrame:
    return load_ads_records(path, source_type=source_type).reset_index(drop=True)


def _sample_indices(total: int, target: int) -> list[int]:
    if total <= 0:
        return []
    sample_n = min(int(max(1, target)), total)
    if sample_n >= total:
        return list(range(total))
    raw = np.linspace(0, total - 1, num=sample_n)
    seen: set[int] = set()
    out: list[int] = []
    for value in raw:
        idx = min(total - 1, max(0, int(round(float(value)))))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if len(out) < sample_n:
        for idx in range(total):
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
            if len(out) >= sample_n:
                break
    return sorted(out)


def _sample_sources(
    publications: pd.DataFrame,
    references: pd.DataFrame | None,
    *,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    pub = publications.copy()
    pub["_compat_source"] = "publications"
    pub["_compat_source_order"] = np.arange(len(pub), dtype=np.int64)

    frames = [pub]
    if references is not None:
        ref = references.copy()
        ref["_compat_source"] = "references"
        ref["_compat_source_order"] = np.arange(len(ref), dtype=np.int64)
        frames.append(ref)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = combined.reset_index(drop=True)
    indices = _sample_indices(len(combined), sample_size)
    sampled = combined.iloc[indices].copy().reset_index(drop=True) if indices else combined.iloc[0:0].copy()

    sampled_publications = (
        sampled[sampled["_compat_source"] == "publications"].drop(columns=["_compat_source", "_compat_source_order"])
        if len(sampled)
        else pub.iloc[0:0].copy()
    )
    sampled_references = None
    if references is not None:
        sampled_references = (
            sampled[sampled["_compat_source"] == "references"].drop(columns=["_compat_source", "_compat_source_order"])
            if len(sampled)
            else references.iloc[0:0].copy()
        )

    sample_meta = {
        "requested_sample_size": int(max(1, sample_size)),
        "sampled_records": int(len(sampled)),
        "sampled_publications": int(len(sampled_publications)),
        "sampled_references": 0 if sampled_references is None else int(len(sampled_references)),
        "source_records_total": int(len(combined)),
    }
    return sampled_publications.reset_index(drop=True), (
        None if sampled_references is None else sampled_references.reset_index(drop=True)
    ), sample_meta


def _texts_from_frame(frame: pd.DataFrame) -> list[str]:
    if len(frame) == 0:
        return []
    titles = frame["title"].fillna("").astype(str).tolist()
    abstracts = frame["abstract"].fillna("").astype(str).tolist()
    from author_name_disambiguation.embedding_contract import build_source_text

    return [build_source_text(title, abstract) for title, abstract in zip(titles, abstracts)]


def _standardize_sample_frame(frame: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    embeddings = [row.astype(np.float32, copy=False).tolist() for row in vectors]
    return pd.DataFrame(
        {
            "Bibcode": frame["bibcode"].astype(str).tolist(),
            "Author": frame["authors"].tolist(),
            "Title_en": frame["title"].tolist(),
            "Abstract_en": frame["abstract"].tolist(),
            "Year": frame["year"].tolist(),
            "Affiliation": frame["aff"].tolist(),
            "precomputed_embedding": embeddings,
        }
    )


def _embed_texts_via_hf(
    *,
    texts: list[str],
    provider: str,
    model_name: str,
    hf_token_env_var: str,
    batch_size: int,
    progress: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = normalize_model_name(model_name)
    vectors, meta = embed_texts_via_hf_httpx(
        texts=texts,
        provider=normalize_provider(provider),
        model_name=model_name,
        hf_token_env_var=hf_token_env_var,
        max_length=256,
        chunk_size=max(1, int(batch_size)),
        progress=bool(progress),
        progress_label="HF compatibility probe",
    )
    return vectors.astype(np.float32, copy=False), {
        "batches_total": int(np.ceil(len(texts) / max(1, int(batch_size)))) if texts else 0,
        "batch_reports": [dict(meta)],
    }


def _cosine_compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for cosine comparison: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.size == 0:
        return {
            "sample_count": 0,
            "shape_local": list(a.shape),
            "shape_hf": list(b.shape),
            "mean_cosine_similarity": None,
            "min_cosine_similarity": None,
            "mean_abs_diff": None,
            "max_abs_diff": None,
        }
    a_norm = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    cosines = ((a / a_norm) * (b / b_norm)).sum(axis=1)
    abs_diff = np.abs(a - b)
    return {
        "sample_count": int(a.shape[0]),
        "shape_local": [int(v) for v in a.shape],
        "shape_hf": [int(v) for v in b.shape],
        "mean_cosine_similarity": float(np.mean(cosines)),
        "min_cosine_similarity": float(np.min(cosines)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "non_finite_local_count": int(np.size(a) - np.isfinite(a).sum()),
        "non_finite_hf_count": int(np.size(b) - np.isfinite(b).sum()),
    }


def _read_assignment_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if len(frame) == 0:
        return frame
    return frame[["mention_id", "author_uid"]].astype(str).sort_values("mention_id").reset_index(drop=True)


def _canonicalize_cluster_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=["mention_id", "cluster_key"])
    grouped = (
        frame.groupby("author_uid", sort=False)["mention_id"]
        .apply(lambda values: tuple(sorted(values.astype(str).tolist())))
        .reset_index(name="cluster_members")
    )
    grouped["cluster_key"] = grouped["cluster_members"].map(lambda members: "||".join(members))
    out = frame.merge(grouped[["author_uid", "cluster_key"]], on="author_uid", how="left")
    return out[["mention_id", "cluster_key"]].astype(str).sort_values("mention_id").reset_index(drop=True)


def _compare_infer_outputs(local_result, hf_result) -> dict[str, Any]:
    local_clusters = _read_assignment_snapshot(local_result.mention_clusters_path)
    hf_clusters = _read_assignment_snapshot(hf_result.mention_clusters_path)
    local_canonical = _canonicalize_cluster_assignments(local_clusters)
    hf_canonical = _canonicalize_cluster_assignments(hf_clusters)
    merged = local_canonical.merge(
        hf_canonical,
        on="mention_id",
        how="outer",
        suffixes=("_local", "_hf"),
        indicator=True,
    )
    changed_assignments = int(
        (
            merged["_merge"].eq("both")
            & merged["cluster_key_local"].notna()
            & merged["cluster_key_hf"].notna()
            & merged["cluster_key_local"].ne(merged["cluster_key_hf"])
        ).sum()
    )
    missing_mentions = int((merged["_merge"] != "both").sum())
    local_clusters_count = int(local_clusters["author_uid"].nunique()) if len(local_clusters) else 0
    hf_clusters_count = int(hf_clusters["author_uid"].nunique()) if len(hf_clusters) else 0
    passed = (
        local_result.go == hf_result.go
        and int(len(local_clusters)) == int(len(hf_clusters))
        and local_clusters_count == hf_clusters_count
        and changed_assignments == 0
        and missing_mentions == 0
    )
    return {
        "passed": bool(passed),
        "go_local": local_result.go,
        "go_hf": hf_result.go,
        "mention_count_local": int(len(local_clusters)),
        "mention_count_hf": int(len(hf_clusters)),
        "cluster_count_local": int(local_clusters_count),
        "cluster_count_hf": int(hf_clusters_count),
        "changed_assignments": int(changed_assignments),
        "missing_mentions": int(missing_mentions),
    }


def _write_markdown_report(payload: dict[str, Any], path: Path) -> Path:
    raw = payload["raw_vector_probe"]
    smoke = payload["downstream_smoke"]
    mini = payload.get("mini_cpu_infer")
    status = "PASS" if payload["compatible"] else "EXPERIMENTAL"
    lines = [
        "# HF Compatibility Report",
        "",
        f"- Status: `{status}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Bundle: `{payload['model_bundle']}`",
        f"- Sampled records: `{payload['sample']['sampled_records']}`",
        f"- Recommendation: `{payload['recommendation']}`",
        "",
        "## Raw Vector Probe",
        "",
        f"- Mean cosine similarity: `{raw['mean_cosine_similarity']}`",
        f"- Min cosine similarity: `{raw['min_cosine_similarity']}`",
        f"- Mean abs diff: `{raw['mean_abs_diff']}`",
        f"- Max abs diff: `{raw['max_abs_diff']}`",
        "",
        "## Downstream Smoke",
        "",
        f"- Passed: `{smoke['passed']}`",
        f"- GO parity: `{smoke['go_local']}` vs `{smoke['go_hf']}`",
        f"- Mention count parity: `{smoke['mention_count_local']}` vs `{smoke['mention_count_hf']}`",
        f"- Cluster count parity: `{smoke['cluster_count_local']}` vs `{smoke['cluster_count_hf']}`",
        f"- Changed assignments: `{smoke['changed_assignments']}`",
    ]
    if mini is not None:
        lines.extend(
            [
                "",
                "## Mini CPU Infer",
                "",
                f"- Wall seconds: `{mini['wall_seconds']}`",
                f"- GO: `{mini['go']}`",
                f"- Mention count: `{mini['mention_count']}`",
                f"- Cluster count: `{mini['cluster_count']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_hf_compatibility_report(request: HfCompatibilityReportRequest) -> HfCompatibilityReportResult:
    run_id = default_run_id("hf_compatibility", tag="cli")
    output_root = _resolved_path(request.output_root)
    report_json_path = output_root / "hf_compatibility_report.json"
    report_markdown_path = output_root / "hf_compatibility_report.md"
    if output_root.exists() and not request.force and (report_json_path.exists() or report_markdown_path.exists()):
        raise FileExistsError(
            "Refusing to overwrite compatibility report artifacts without force=True: "
            f"{report_json_path}, {report_markdown_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    model_info = _resolve_model_bundle(request.model_bundle)
    bundle_contract = dict(model_info["manifest"].get("embedding_contract") or build_bundle_embedding_contract(model_info["model_cfg"]))
    text_contract = dict(bundle_contract.get("text", {}) or {})
    model_name = normalize_model_name(request.model_name)
    if text_contract.get("model_name") and str(text_contract["model_name"]) != model_name:
        raise ValueError(
            "HF compatibility report must use the bundle text embedding contract. "
            f"bundle={text_contract.get('model_name')!r} request={model_name!r}"
        )

    publications = _load_normalized_source(request.publications_path, source_type="publication")
    references = (
        None if request.references_path is None else _load_normalized_source(request.references_path, source_type="reference")
    )
    sample_publications, sample_references, sample_meta = _sample_sources(
        publications,
        references,
        sample_size=int(request.sample_size),
    )
    combined_sample = pd.concat(
        [frame for frame in [sample_publications, sample_references] if frame is not None and len(frame) > 0],
        ignore_index=True,
    )
    if len(combined_sample) == 0:
        raise RuntimeError("Compatibility report sample is empty; provide a non-empty source dataset.")

    local_vectors, local_meta = generate_specter_embeddings(
        mentions=combined_sample,
        model_name=model_name,
        text_backend=text_contract.get("text_backend", "transformers"),
        text_adapter_name=text_contract.get("text_adapter_name"),
        text_adapter_alias=text_contract.get("text_adapter_alias", "specter2"),
        max_length=int(text_contract.get("tokenization", {}).get("max_length", 256)),
        batch_size=int(request.batch_size),
        device=str(request.device),
        precision_mode="fp32",
        prefer_precomputed=False,
        use_stub_if_missing=False,
        show_progress=bool(request.progress),
        quiet_libraries=True,
        reuse_model=False,
        return_meta=True,
    )
    hf_vectors, hf_meta = _embed_texts_via_hf(
        texts=_texts_from_frame(combined_sample),
        provider=request.provider,
        model_name=model_name,
        hf_token_env_var=request.hf_token_env_var,
        batch_size=int(request.batch_size),
        progress=bool(request.progress),
    )
    raw_probe = _cosine_compare(local_vectors, hf_vectors)

    pub_count = int(len(sample_publications))
    local_publications = _standardize_sample_frame(sample_publications, local_vectors[:pub_count])
    hf_publications = _standardize_sample_frame(sample_publications, hf_vectors[:pub_count])

    local_references = None
    hf_references = None
    if sample_references is not None:
        local_references = _standardize_sample_frame(sample_references, local_vectors[pub_count:])
        hf_references = _standardize_sample_frame(sample_references, hf_vectors[pub_count:])

    local_dataset_root = output_root / "sample_local"
    hf_dataset_root = output_root / "sample_hf"
    local_dataset_root.mkdir(parents=True, exist_ok=True)
    hf_dataset_root.mkdir(parents=True, exist_ok=True)

    local_publications_path = local_dataset_root / "publications.parquet"
    hf_publications_path = hf_dataset_root / "publications.parquet"
    local_publications.to_parquet(local_publications_path, index=False)
    hf_publications.to_parquet(hf_publications_path, index=False)

    local_references_path = None
    hf_references_path = None
    if local_references is not None and hf_references is not None:
        local_references_path = local_dataset_root / "references.parquet"
        hf_references_path = hf_dataset_root / "references.parquet"
        local_references.to_parquet(local_references_path, index=False)
        hf_references.to_parquet(hf_references_path, index=False)

    smoke_local = run_infer_sources(
        InferSourcesRequest(
            publications_path=local_publications_path,
            references_path=local_references_path,
            output_root=output_root / "infer_smoke_local",
            dataset_id=f"{request.dataset_id}__compat_local",
            model_bundle=request.model_bundle,
            infer_stage="smoke",
            device="cpu",
            precision_mode="fp32",
            cluster_backend="sklearn_cpu",
            force=True,
            progress=False,
        )
    )
    smoke_hf = run_infer_sources(
        InferSourcesRequest(
            publications_path=hf_publications_path,
            references_path=hf_references_path,
            output_root=output_root / "infer_smoke_hf",
            dataset_id=f"{request.dataset_id}__compat_hf",
            model_bundle=request.model_bundle,
            infer_stage="smoke",
            device="cpu",
            precision_mode="fp32",
            cluster_backend="sklearn_cpu",
            force=True,
            progress=False,
        )
    )
    smoke_compare = _compare_infer_outputs(smoke_local, smoke_hf)

    mini_payload = None
    if smoke_compare["passed"]:
        mini_started_at = perf_counter()
        mini_result = run_infer_sources(
            InferSourcesRequest(
                publications_path=hf_publications_path,
                references_path=hf_references_path,
                output_root=output_root / "infer_mini_hf",
                dataset_id=f"{request.dataset_id}__compat_hf_mini",
                model_bundle=request.model_bundle,
                infer_stage="mini",
                device="cpu",
                precision_mode="fp32",
                cluster_backend="sklearn_cpu",
                force=True,
                progress=False,
            )
        )
        mini_clusters = pd.read_parquet(mini_result.mention_clusters_path)
        mini_payload = {
            "wall_seconds": float(perf_counter() - mini_started_at),
            "go": mini_result.go,
            "mention_count": int(len(mini_clusters)),
            "cluster_count": int(mini_clusters["author_uid"].nunique()) if len(mini_clusters) else 0,
            "stage_metrics_path": str(mini_result.stage_metrics_path),
        }

    compatible = bool(smoke_compare["passed"])
    recommendation = (
        "Promote HF remote SPECTER for the current bundle."
        if compatible
        else "Keep HF remote SPECTER experimental until parity is restored."
    )
    payload = {
        "run_id": run_id,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset_id": str(request.dataset_id),
        "model_bundle": str(_resolved_path(request.model_bundle)),
        "compatible": compatible,
        "recommendation": recommendation,
        "embedding_contract": bundle_contract,
        "sample": sample_meta,
        "raw_vector_probe": {
            **raw_probe,
            "local_runtime": local_meta,
            "hf_runtime": hf_meta,
        },
        "downstream_smoke": smoke_compare,
        "mini_cpu_infer": mini_payload,
        "artifacts": {
            "output_root": str(output_root),
            "report_json_path": str(report_json_path),
            "report_markdown_path": str(report_markdown_path),
            "sample_local_publications_path": str(local_publications_path),
            "sample_hf_publications_path": str(hf_publications_path),
            "sample_local_references_path": None if local_references_path is None else str(local_references_path),
            "sample_hf_references_path": None if hf_references_path is None else str(hf_references_path),
        },
    }
    write_json(payload, report_json_path)
    _write_markdown_report(payload, report_markdown_path)
    return HfCompatibilityReportResult(
        run_id=run_id,
        output_root=output_root,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        compatible=compatible,
    )
