from __future__ import annotations

from pathlib import Path

from author_name_disambiguation.common.package_resources import resource_path


FIXED_MODEL_BASELINE_RUN_ID = "full_20260218T111506Z_cli02681429"
FIXED_MODEL_BUNDLE_RESOURCE = "resources/model_bundles/fixed_model_baseline/bundle_v1"
DEFAULT_RAW_LSPO_PARQUET = Path("data/raw/lspo/LSPO_v1.parquet")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_ARTIFACTS_ROOT = Path("artifacts")


def resolve_fixed_model_bundle_path() -> Path:
    return resource_path(FIXED_MODEL_BUNDLE_RESOURCE)

