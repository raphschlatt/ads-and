from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_release_artifacts_are_inference_only(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(dist_dir.glob("ads_and-0.2.0-*.whl"))
    sdist_path = next(dist_dir.glob("ads_and-0.2.0.tar.gz"))

    assert wheel_path.stat().st_size < 6_000_000

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_members = set(wheel.namelist())
    assert "author_name_disambiguation/public_cli.py" in wheel_members
    assert "author_name_disambiguation/public_api.py" in wheel_members
    assert (
        "author_name_disambiguation/resources/model_bundles/fixed_model_baseline/bundle_v1/checkpoint.pt"
        in wheel_members
    )
    assert "author_name_disambiguation/cli.py" not in wheel_members
    assert "author_name_disambiguation/api.py" not in wheel_members
    assert "author_name_disambiguation/precompute_source_embeddings.py" not in wheel_members
    assert "author_name_disambiguation/approaches/nand/train.py" not in wheel_members
    assert "author_name_disambiguation/resources/models/nand_best.yaml" not in wheel_members
    assert "author_name_disambiguation/resources/train_runs/full.yaml" not in wheel_members
    assert "author_name_disambiguation_research/__init__.py" not in wheel_members

    with tarfile.open(sdist_path, "r:gz") as sdist:
        sdist_members = set(sdist.getnames())
    root_prefix = f"ads_and-0.2.0"
    assert f"{root_prefix}/README.md" in sdist_members
    assert f"{root_prefix}/LICENSE" in sdist_members
    assert f"{root_prefix}/CITATION.cff" in sdist_members
    assert f"{root_prefix}/docs/inference_workflow.md" not in sdist_members
    assert f"{root_prefix}/src/author_name_disambiguation/cli.py" not in sdist_members
    assert f"{root_prefix}/src/author_name_disambiguation/api.py" not in sdist_members
    assert f"{root_prefix}/src/author_name_disambiguation/approaches/nand/train.py" not in sdist_members
    assert f"{root_prefix}/author_name_disambiguation_research/__init__.py" not in sdist_members
