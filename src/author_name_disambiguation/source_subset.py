from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from author_name_disambiguation.common.io_schema import save_parquet
from author_name_disambiguation.data.prepare_ads import load_ads_records
from author_name_disambiguation.embedding_contract import CANONICAL_TEXT_EMBEDDING_FIELD


@dataclass(slots=True)
class SourceSubsetResult:
    output_root: Path
    publications_path: Path
    references_path: Path | None
    publications_rows: int
    references_rows: int


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _standardize_subset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Bibcode": frame["bibcode"].astype(str).tolist() if "bibcode" in frame.columns else [],
            "Author": frame["authors"].tolist() if "authors" in frame.columns else [],
            "Title_en": frame["title"].tolist() if "title" in frame.columns else [],
            "Abstract_en": frame["abstract"].tolist() if "abstract" in frame.columns else [],
            "Year": frame["year"].tolist() if "year" in frame.columns else [],
            "Affiliation": frame["aff"].tolist() if "aff" in frame.columns else [],
        }
    )
    if CANONICAL_TEXT_EMBEDDING_FIELD in frame.columns:
        out[CANONICAL_TEXT_EMBEDDING_FIELD] = frame[CANONICAL_TEXT_EMBEDDING_FIELD].tolist()
    return out


def build_contract_valid_source_subset(
    *,
    publications_path: str | Path,
    output_root: str | Path,
    references_path: str | Path | None = None,
    max_publications: int = 32,
    max_references: int = 32,
    drop_precomputed_embeddings: bool = True,
) -> SourceSubsetResult:
    resolved_output_root = _resolved_path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    publications = load_ads_records(_resolved_path(publications_path), source_type="publication").reset_index(drop=True)
    publications_subset = publications.head(max(0, int(max_publications))).copy()
    if drop_precomputed_embeddings and CANONICAL_TEXT_EMBEDDING_FIELD in publications_subset.columns:
        publications_subset = publications_subset.drop(columns=[CANONICAL_TEXT_EMBEDDING_FIELD])
    publications_output_path = resolved_output_root / "publications_subset.parquet"
    save_parquet(_standardize_subset_frame(publications_subset), publications_output_path, index=False)

    references_output_path: Path | None = None
    references_rows = 0
    if references_path is not None:
        references = load_ads_records(_resolved_path(references_path), source_type="reference").reset_index(drop=True)
        references_subset = references.head(max(0, int(max_references))).copy()
        if drop_precomputed_embeddings and CANONICAL_TEXT_EMBEDDING_FIELD in references_subset.columns:
            references_subset = references_subset.drop(columns=[CANONICAL_TEXT_EMBEDDING_FIELD])
        references_output_path = resolved_output_root / "references_subset.parquet"
        save_parquet(_standardize_subset_frame(references_subset), references_output_path, index=False)
        references_rows = int(len(references_subset))

    return SourceSubsetResult(
        output_root=resolved_output_root,
        publications_path=publications_output_path,
        references_path=references_output_path,
        publications_rows=int(len(publications_subset)),
        references_rows=int(references_rows),
    )
