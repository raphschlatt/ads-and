from __future__ import annotations

import author_name_disambiguation_research as research
from author_name_disambiguation_research import api as research_api
from author_name_disambiguation_research import cli as research_cli


def test_research_workspace_exports_repo_only_api():
    assert hasattr(research, "evaluate_lspo_quality")
    assert hasattr(research, "train_lspo_model")
    assert research_api.evaluate_lspo_quality is research.evaluate_lspo_quality
    assert callable(research_cli.main)
