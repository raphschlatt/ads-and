from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "ops" / "compare_cluster_test_reports.py"


def _write_report(path: Path, *, run_id: str, seeds: list[int], status: str, f1: float, precision: float) -> None:
    payload = {
        "model_run_id": run_id,
        "status": status,
        "seeds_evaluated": seeds,
        "variants": {
            "dbscan_with_constraints": {
                "f1_mean": float(f1),
                "precision_mean": float(precision),
                "recall_mean": 0.97,
                "accuracy_mean": 0.94,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_cluster_test_reports_passes_gate(tmp_path: Path):
    baseline = tmp_path / "baseline_06.json"
    candidate = tmp_path / "candidate_06.json"
    _write_report(baseline, run_id="base_run", seeds=[1, 2, 3, 4, 5], status="ok", f1=0.9700, precision=0.9635)
    _write_report(candidate, run_id="cand_run", seeds=[1, 2, 3, 4, 5], status="ok", f1=0.9720, precision=0.9630)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    out_path = candidate.parent / "99_compare_cluster_report_to_baseline.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["decision"] == "promote_candidate"
    assert payload["metrics"]["delta"]["f1_mean"] > 0.0
    assert payload["metrics"]["delta"]["precision_mean"] >= -0.001


def test_compare_cluster_test_reports_fails_on_precision_drop(tmp_path: Path):
    baseline = tmp_path / "baseline_06.json"
    candidate = tmp_path / "candidate_06.json"
    _write_report(baseline, run_id="base_run", seeds=[1, 2, 3], status="ok", f1=0.9700, precision=0.9635)
    _write_report(candidate, run_id="cand_run", seeds=[1, 2, 3], status="ok", f1=0.9710, precision=0.9600)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    out_path = candidate.parent / "99_compare_cluster_report_to_baseline.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert any("delta_precision_mean" in item for item in payload["failures"])


def test_compare_cluster_test_reports_fails_on_seed_mismatch(tmp_path: Path):
    baseline = tmp_path / "baseline_06.json"
    candidate = tmp_path / "candidate_06.json"
    _write_report(baseline, run_id="base_run", seeds=[1, 2, 3], status="ok", f1=0.9700, precision=0.9635)
    _write_report(candidate, run_id="cand_run", seeds=[1, 2], status="ok", f1=0.9720, precision=0.9636)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    out_path = candidate.parent / "99_compare_cluster_report_to_baseline.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert any("seeds mismatch" in item for item in payload["failures"])
