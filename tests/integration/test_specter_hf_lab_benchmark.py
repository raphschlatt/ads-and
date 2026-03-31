from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from author_name_disambiguation import cli


def test_cli_run_specter_hf_lab_benchmark_writes_reports_and_marks_unavailable_modes(
    monkeypatch,
    tmp_path: Path,
):
    lab_module = __import__("author_name_disambiguation.specter_hf_lab_benchmark", fromlist=["dummy"])

    monkeypatch.setattr(
        lab_module,
        "_resolve_model_bundle",
        lambda _bundle: {
            "manifest": {
                "embedding_contract": {
                    "text": {
                        "model_name": "allenai/specter",
                        "tokenization": {"max_length": 256},
                    }
                }
            },
            "model_cfg": {"representation": {"text_model_name": "allenai/specter", "max_length": 256}},
        },
    )
    monkeypatch.setattr(lab_module, "_collect_hardware_metadata", lambda: {"gpu_name": "Mock GPU"})
    monkeypatch.setattr(
        lab_module,
        "_build_micro_short_repeat_dataset",
        lambda repeat_count: lab_module._LabDataset(
            name="micro_short_repeat",
            texts=["short text"] * int(repeat_count),
            manifest=[{"sample_index": idx} for idx in range(int(repeat_count))],
        ),
    )
    monkeypatch.setattr(
        lab_module,
        "_build_ads_realistic_truncated_dataset",
        lambda **_kwargs: lab_module._LabDataset(
            name="ads_realistic_truncated",
            texts=["ads text a", "ads text b"],
            manifest=[{"sample_index": 0}, {"sample_index": 1}],
        ),
    )
    monkeypatch.setattr(
        lab_module,
        "_select_mode_specs",
        lambda _profiles: [
            lab_module._LabModeSpec("hf_sync_hub_client_baseline", transport="sync_hub", lab_only=False, non_production=False),
            lab_module._LabModeSpec("hf_httpx_async_pool_prod_safe", transport="httpx", lab_only=True, non_production=False),
            lab_module._LabModeSpec("hf_aiohttp_pool_prod_safe", transport="aiohttp", lab_only=True, non_production=False),
        ],
    )

    def _fake_run_mode_variant(*, dataset, mode_spec, concurrency=None, **_kwargs):
        if mode_spec.transport == "aiohttp":
            raise RuntimeError("missing aiohttp")
        texts_total = len(dataset.texts)
        vectors = np.full((texts_total, 768), 1.0 if mode_spec.transport == "sync_hub" else 2.0, dtype=np.float32)
        return lab_module._LabRun(
            available=True,
            vectors=vectors,
            success_mask=np.ones((texts_total,), dtype=bool),
            errors=[None] * texts_total,
            raw_shapes=["[32, 768]"] * texts_total,
            warmup_seconds=0.1,
            processing_wall_seconds=2.0 if mode_spec.transport == "sync_hub" else 1.0,
            meta={
                "transport": mode_spec.transport,
                "concurrency": concurrency,
                "lab_only": bool(mode_spec.lab_only),
                "non_production": bool(mode_spec.non_production),
            },
        )

    monkeypatch.setattr(lab_module, "_run_mode_variant", _fake_run_mode_variant)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-specter-hf-lab-benchmark",
            "--publications-path",
            str(tmp_path / "publications.parquet"),
            "--output-root",
            str(tmp_path / "hf_lab"),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(tmp_path / "bundle"),
            "--profiles",
            "all",
            "--concurrency-values",
            "4,8",
            "--micro-repeat-count",
            "4",
            "--realistic-sample-size",
            "2",
            "--no-progress",
        ]
    )
    payload = args.func(args)

    report_json = json.loads((tmp_path / "hf_lab" / "specter_hf_lab_report.json").read_text(encoding="utf-8"))
    report_md = (tmp_path / "hf_lab" / "specter_hf_lab_report.md").read_text(encoding="utf-8")

    assert payload["summary"] == report_json["decision"]["summary"]
    assert report_json["datasets"]["micro_short_repeat"]["best_speedup_vs_sync"] == 2.0
    assert report_json["datasets"]["ads_realistic_truncated"]["modes"]["hf_aiohttp_pool_prod_safe"]["available"] is False
    assert (
        report_json["datasets"]["ads_realistic_truncated"]["modes"]["hf_aiohttp_pool_prod_safe"]["error"]
        == "missing aiohttp"
    )
    assert "SPECTER HF Lab Benchmark Report" in report_md


def test_run_specter_hf_lab_benchmark_cleans_empty_output_root_on_failure(monkeypatch, tmp_path: Path):
    lab_module = __import__("author_name_disambiguation.specter_hf_lab_benchmark", fromlist=["dummy"])
    monkeypatch.setattr(lab_module, "_resolve_model_bundle", lambda _bundle: (_ for _ in ()).throw(RuntimeError("boom")))

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-specter-hf-lab-benchmark",
            "--publications-path",
            str(tmp_path / "publications.parquet"),
            "--output-root",
            str(tmp_path / "hf_lab_fail"),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(tmp_path / "bundle"),
            "--no-progress",
        ]
    )

    try:
        args.func(args)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError")

    assert not (tmp_path / "hf_lab_fail").exists()
