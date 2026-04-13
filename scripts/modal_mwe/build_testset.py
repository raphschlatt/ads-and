from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_testset(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    publications_path = root / "publications.parquet"
    references_path = root / "references.parquet"

    pd.DataFrame(
        [
            {
                "Bibcode": "bib1",
                "Author": ["Doe J", "Roe A"],
                "Title_en": "Paper 1",
                "Abstract_en": "Abstract 1",
                "Year": 2020,
                "Affiliation": ["Inst A", "Inst B"],
                "DOI": "10.1/test-1",
            },
            {
                "Bibcode": "bib2",
                "Author": ["Doe J."],
                "Title_en": "Paper 2",
                "Abstract_en": "Abstract 2",
                "Year": 2021,
                "Affiliation": ["Inst C"],
                "Keywords": ["kw-a"],
            },
        ]
    ).to_parquet(publications_path, index=False)

    pd.DataFrame(
        [
            {
                "Bibcode": "bib3",
                "Author": ["Ref X", "Ref Y"],
                "Title_en": "Paper 3",
                "Abstract_en": "Abstract 3",
                "Year": 2022,
                "Affiliation": ["Inst R1", "Inst R2"],
                "PDF_URL": "https://example.test/ref-3.pdf",
            }
        ]
    ).to_parquet(references_path, index=False)

    return {
        "root": root,
        "publications_path": publications_path,
        "references_path": references_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the tiny standalone Modal MWE testset.")
    parser.add_argument(
        "--output-dir",
        default="tmp/modal_mwe_smoke",
        help="Directory for the generated tiny publications/references parquet files",
    )
    args = parser.parse_args()
    result = build_testset(args.output_dir)
    print(result["root"])


if __name__ == "__main__":
    main()
