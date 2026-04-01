from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from author_name_disambiguation.common.io_schema import save_parquet
from author_name_disambiguation.common.pipeline_reports import default_run_id, write_json
from author_name_disambiguation.data.prepare_ads import load_ads_records
from author_name_disambiguation.embedding_contract import (
    CANONICAL_TEXT_EMBEDDING_FIELD,
    DEFAULT_TEXT_MODEL_NAME,
    TEXT_EMBEDDING_DIM,
    build_bundle_embedding_contract,
    build_source_text,
)
from author_name_disambiguation.features.embed_specter import _coerce_precomputed_embedding
from author_name_disambiguation.hf_transport import (
    DEFAULT_HF_TOKEN_ENV_VAR,
    embed_texts_via_hf_endpoint,
    normalize_model_name,
    resolve_hf_token,
)


@dataclass(slots=True)
class PrecomputeSourceEmbeddingsRequest:
    publications_path: str | Path
    output_root: str | Path
    references_path: str | Path | None = None
    dataset_id: str | None = None
    hf_token_env_var: str = DEFAULT_HF_TOKEN_ENV_VAR
    force: bool = False
    progress: bool = True


@dataclass(slots=True)
class PrecomputeSourceEmbeddingsResult:
    run_id: str
    output_root: Path
    publications_output_path: Path
    references_output_path: Path | None
    report_path: Path


@dataclass(slots=True)
class _LoadedSource:
    label: str
    input_path: Path
    normalized: pd.DataFrame
    raw_source: pd.DataFrame | None
    output_path: Path
    load_meta: dict[str, Any]


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _normalize_model_name(model_name: str) -> str:
    return normalize_model_name(model_name)


def _resolve_hf_token(env_var_name: str) -> str:
    return resolve_hf_token(env_var_name)


def _load_source(
    *,
    label: str,
    path: str | Path,
    output_root: Path,
) -> _LoadedSource:
    input_path = _resolved_path(path)
    normalized, raw_source, load_meta = load_ads_records(
        input_path,
        source_type="publication" if label == "publications" else "reference",
        return_raw_source=True,
        return_meta=True,
    )
    output_path = output_root / f"{label}_precomputed.parquet"
    return _LoadedSource(
        label=str(label),
        input_path=input_path,
        normalized=normalized.reset_index(drop=True),
        raw_source=raw_source,
        output_path=output_path,
        load_meta=dict(load_meta or {}),
    )


def _build_record_texts(frame: pd.DataFrame) -> list[str]:
    titles = frame["title"].fillna("").astype(str).tolist() if "title" in frame.columns else []
    abstracts = frame["abstract"].fillna("").astype(str).tolist() if "abstract" in frame.columns else []
    return [build_source_text(title, abstract) for title, abstract in zip(titles, abstracts)]


def _standardize_output_frame(frame: pd.DataFrame, embeddings: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Bibcode": frame["bibcode"].astype(str).tolist() if "bibcode" in frame.columns else [],
            "Author": frame["authors"].tolist() if "authors" in frame.columns else [],
            "Title_en": frame["title"].tolist() if "title" in frame.columns else [],
            "Abstract_en": frame["abstract"].tolist() if "abstract" in frame.columns else [],
            "Year": frame["year"].tolist() if "year" in frame.columns else [],
            "Affiliation": frame["aff"].tolist() if "aff" in frame.columns else [],
            CANONICAL_TEXT_EMBEDDING_FIELD: embeddings,
        }
    )


def _build_output_frame(source: _LoadedSource, embeddings: list[list[float]]) -> pd.DataFrame:
    if source.raw_source is None:
        return _standardize_output_frame(source.normalized, embeddings)

    out = source.raw_source.copy()
    if CANONICAL_TEXT_EMBEDDING_FIELD not in out.columns:
        out[CANONICAL_TEXT_EMBEDDING_FIELD] = [None] * len(out)
    else:
        out[CANONICAL_TEXT_EMBEDDING_FIELD] = out[CANONICAL_TEXT_EMBEDDING_FIELD].astype(object)

    source_indices = source.normalized["source_row_idx"].astype(int).tolist() if len(source.normalized) else []
    for source_row_idx, vector in zip(source_indices, embeddings):
        if 0 <= int(source_row_idx) < len(out):
            out.at[int(source_row_idx), CANONICAL_TEXT_EMBEDDING_FIELD] = vector
    return out


def _compute_vectors_for_source_records(
    *,
    sources: list[_LoadedSource],
    request: PrecomputeSourceEmbeddingsRequest,
) -> tuple[dict[str, list[list[float]]], dict[str, Any]]:
    text_to_vector: dict[str, list[float]] = {}
    dataset_vectors: dict[str, list[list[float]]] = {}
    dataset_stats: dict[str, Any] = {}
    missing_texts: list[str] = []

    for source in sources:
        texts = _build_record_texts(source.normalized)
        existing_values = (
            source.normalized[CANONICAL_TEXT_EMBEDDING_FIELD].tolist()
            if CANONICAL_TEXT_EMBEDDING_FIELD in source.normalized.columns
            else []
        )
        rows: list[list[float] | None] = []
        reused_count = 0
        empty_text_count = 0
        missing_count = 0
        unique_missing_texts: list[str] = []
        seen_missing_texts: set[str] = set()

        for text, existing in zip(texts, existing_values if existing_values else [None] * len(texts)):
            cached = _coerce_precomputed_embedding(existing, dim=TEXT_EMBEDDING_DIM)
            if cached is not None:
                rows.append(cached.astype(np.float32, copy=False).tolist())
                reused_count += 1
                continue

            if not text:
                empty_text_count += 1
            rows.append(None)
            missing_count += 1
            if text not in seen_missing_texts:
                seen_missing_texts.add(text)
                unique_missing_texts.append(text)

        dataset_vectors[source.label] = rows  # type: ignore[assignment]
        dataset_stats[source.label] = {
            "rows": int(len(source.normalized)),
            "reused_precomputed_count": int(reused_count),
            "missing_precomputed_count": int(missing_count),
            "empty_text_count": int(empty_text_count),
            "unique_missing_text_count": int(len(unique_missing_texts)),
        }
        missing_texts.extend(unique_missing_texts)

    unique_missing_texts_all = list(dict.fromkeys(missing_texts))
    remote_meta: dict[str, Any] = {
        "transport": "hf_endpoint",
        "runtime_backend": "hf_endpoint",
        "generation_mode": "remote_endpoint_only",
        "texts_total": 0,
        "texts_successful": 0,
        "texts_failed": 0,
    }
    if unique_missing_texts_all:
        vectors, remote_meta = embed_texts_via_hf_endpoint(
            texts=unique_missing_texts_all,
            model_name=DEFAULT_TEXT_MODEL_NAME,
            hf_token_env_var=request.hf_token_env_var,
            max_length=256,
            progress=bool(request.progress),
            progress_label="HF endpoint source embeddings",
        )
        for text, vector in zip(unique_missing_texts_all, vectors, strict=True):
            text_to_vector[text] = vector.astype(np.float32, copy=False).tolist()

    resolved_vectors: dict[str, list[list[float]]] = {}
    for source in sources:
        texts = _build_record_texts(source.normalized)
        rows = dataset_vectors[source.label]
        resolved_rows: list[list[float]] = []
        for text, row in zip(texts, rows, strict=True):
            if row is not None:
                resolved_rows.append(list(row))
                continue
            vector = text_to_vector.get(text)
            if vector is None:
                raise RuntimeError(f"Missing remote embedding for source text in {source.label!r}.")
            resolved_rows.append(list(vector))
        resolved_vectors[source.label] = resolved_rows

    hf_runtime = {
        "model_name": DEFAULT_TEXT_MODEL_NAME,
        "texts_remote_computed": int(len(unique_missing_texts_all)),
        **remote_meta,
    }
    return resolved_vectors, {"datasets": dataset_stats, "hf_runtime": hf_runtime}


def precompute_source_embeddings(request: PrecomputeSourceEmbeddingsRequest) -> PrecomputeSourceEmbeddingsResult:
    run_id = default_run_id("precompute_source_embeddings", tag="cli")
    output_root = _resolved_path(request.output_root)
    publications_output_path = output_root / "publications_precomputed.parquet"
    references_output_path = output_root / "references_precomputed.parquet" if request.references_path is not None else None
    report_path = output_root / "precompute_source_embeddings_report.json"
    output_root.mkdir(parents=True, exist_ok=True)

    existing_outputs = [publications_output_path, report_path]
    if references_output_path is not None:
        existing_outputs.append(references_output_path)
    if not request.force:
        conflicts = [path for path in existing_outputs if path.exists()]
        if conflicts:
            raise FileExistsError(
                "Refusing to overwrite existing precompute artifacts without force=True: "
                + ", ".join(str(path) for path in conflicts)
            )

    started_at = perf_counter()
    sources = [
        _load_source(label="publications", path=request.publications_path, output_root=output_root),
    ]
    if request.references_path is not None:
        sources.append(_load_source(label="references", path=request.references_path, output_root=output_root))

    model_name = _normalize_model_name(DEFAULT_TEXT_MODEL_NAME)

    vectors_by_source, compute_meta = _compute_vectors_for_source_records(
        sources=sources,
        request=PrecomputeSourceEmbeddingsRequest(
            publications_path=request.publications_path,
            references_path=request.references_path,
            output_root=request.output_root,
            dataset_id=request.dataset_id,
            hf_token_env_var=request.hf_token_env_var,
            force=request.force,
            progress=request.progress,
        ),
    )

    output_rows_total = 0
    for source in sources:
        embeddings = vectors_by_source[source.label]
        output_frame = _build_output_frame(source, embeddings)
        save_parquet(output_frame, source.output_path, index=False)
        output_rows_total += int(len(output_frame))

    report_payload = {
        "run_id": run_id,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "pipeline_scope": "precompute_source_embeddings",
        "dataset_id": None if request.dataset_id is None else str(request.dataset_id),
        "model_name": model_name,
        "hf_token_env_var": str(request.hf_token_env_var),
        "embedding_contract": build_bundle_embedding_contract(
            {"representation": {"text_model_name": model_name}}
        )["text"],
        "inputs": {
            "publications_path": str(_resolved_path(request.publications_path)),
            "references_path": None if request.references_path is None else str(_resolved_path(request.references_path)),
        },
        "outputs": {
            "output_root": str(output_root),
            "publications_output_path": str(publications_output_path),
            "references_output_path": None if references_output_path is None else str(references_output_path),
            "report_path": str(report_path),
        },
        "runtime": {
            "wall_seconds": float(perf_counter() - started_at),
            "output_rows_total": int(output_rows_total),
            **dict(compute_meta.get("hf_runtime", {}) or {}),
        },
        "datasets": {},
    }
    for source in sources:
        load_meta = dict(source.load_meta or {})
        stats = dict(compute_meta.get("datasets", {}).get(source.label, {}) or {})
        report_payload["datasets"][source.label] = {
            "input_path": str(source.input_path),
            "output_path": str(source.output_path),
            "input_rows_normalized": int(len(source.normalized)),
            "output_rows_written": int(len(source.raw_source) if source.raw_source is not None else len(source.normalized)),
            "load": load_meta,
            **stats,
        }

    write_json(report_payload, report_path)
    return PrecomputeSourceEmbeddingsResult(
        run_id=run_id,
        output_root=output_root,
        publications_output_path=publications_output_path,
        references_output_path=references_output_path,
        report_path=report_path,
    )
