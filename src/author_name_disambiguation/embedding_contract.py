from __future__ import annotations

from typing import Any, Mapping

TEXT_EMBEDDING_DIM = 768
NAME_EMBEDDING_DIM = 50
DEFAULT_TEXT_MODEL_NAME = "allenai/specter"
DEFAULT_TEXT_BACKEND = "transformers"
DEFAULT_TEXT_MAX_LENGTH = 256
DEFAULT_TEXT_POOLING = "cls_first_token"
CANONICAL_TEXT_EMBEDDING_FIELD = "precomputed_embedding"
LEGACY_TEXT_EMBEDDING_FIELDS = ("embedding",)


def build_source_text(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        return f"{title} [SEP] {abstract}"
    return title or abstract


def build_text_embedding_contract(model_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rep_cfg = dict((model_cfg or {}).get("representation", {}) or {})
    return {
        "family": "specter",
        "provider": "huggingface",
        "field_name": CANONICAL_TEXT_EMBEDDING_FIELD,
        "legacy_field_names": list(LEGACY_TEXT_EMBEDDING_FIELDS),
        "model_name": str(rep_cfg.get("text_model_name", DEFAULT_TEXT_MODEL_NAME)),
        "text_backend": str(rep_cfg.get("text_backend", DEFAULT_TEXT_BACKEND)),
        "text_adapter_name": rep_cfg.get("text_adapter_name"),
        "text_adapter_alias": str(rep_cfg.get("text_adapter_alias", "specter2")),
        "dimension": int(TEXT_EMBEDDING_DIM),
        "text_builder": "title [SEP] abstract",
        "separator": " [SEP] ",
        "title_columns": ["Title_en", "Title", "title"],
        "abstract_columns": ["Abstract_en", "Abstract", "abstract"],
        "pooling": DEFAULT_TEXT_POOLING,
        "tokenization": {
            "truncation": True,
            "max_length": int(rep_cfg.get("max_length", DEFAULT_TEXT_MAX_LENGTH)),
        },
        "compatibility_note": (
            "The active NAND bundle expects SPECTER-compatible document embeddings. "
            "Equal dimensionality alone is not a compatibility guarantee."
        ),
    }


def build_name_embedding_contract() -> dict[str, Any]:
    return {
        "family": "chars2vec",
        "model_name": "eng_50",
        "dimension": int(NAME_EMBEDDING_DIM),
        "execution_mode": "predict",
        "reference_batch_size": 32,
    }


def build_bundle_embedding_contract(model_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "text": build_text_embedding_contract(model_cfg),
        "name": build_name_embedding_contract(),
    }
