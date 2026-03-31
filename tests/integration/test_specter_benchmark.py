from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from author_name_disambiguation import cli


def test_cli_run_specter_benchmark_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    benchmark_module = __import__("author_name_disambiguation.specter_benchmark", fromlist=["dummy"])

    sample_frame = pd.DataFrame(
        [
            {
                "bibcode": "pub1",
                "authors": ["Doe J", "Roe A"],
                "title": "Paper 1",
                "abstract": "Abstract 1",
                "year": 2020,
                "aff": ["Inst A", "Inst B"],
            },
            {
                "bibcode": "pub2",
                "authors": ["Doe J"],
                "title": "Paper 2",
                "abstract": "Abstract 2",
                "year": 2021,
                "aff": ["Inst C"],
            },
        ]
    )

    monkeypatch.setattr(
        benchmark_module,
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
    monkeypatch.setattr(
        benchmark_module,
        "_load_normalized_source",
        lambda _path, source_type: sample_frame.copy() if source_type == "publication" else sample_frame.iloc[0:0].copy(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_build_tokenizer",
        lambda _model_name: object(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_compute_raw_token_counts_for_frame",
        lambda frame, tokenizer: np.asarray([120 + idx * 100 for idx in range(len(frame))], dtype=np.int32),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_compute_raw_token_counts_for_texts",
        lambda texts, tokenizer: np.asarray([213 for _ in texts], dtype=np.int32),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_load_notebook_mwe_text",
        lambda notebook_path=None: {
            "notebook_path": str(Path(notebook_path or tmp_path / "Test.ipynb")),
            "title": "MWE Title",
            "abstract": "MWE Abstract",
            "text": "MWE Title [SEP] MWE Abstract",
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "_collect_hardware_metadata",
        lambda: {"cuda_available": True, "gpu_name": "Mock GPU"},
    )

    class _FakeSession:
        def __init__(self, mode: str):
            self.mode = mode

        def run(self, *, sample, cap, batch_size, progress):
            vectors = np.vstack(
                [np.full((1, 768), fill_value=float(idx + (1 if self.mode == "gpu" else 2)), dtype=np.float32) for idx in range(len(sample.texts))]
            )
            return benchmark_module._ModeRun(
                mode=f"local_{self.mode}",
                available=True,
                vectors=vectors,
                attempted_mask=np.ones((len(sample.texts),), dtype=bool),
                success_mask=np.ones((len(sample.texts),), dtype=bool),
                per_item_wall_seconds=np.full((len(sample.texts),), 0.1 if self.mode == "gpu" else 0.2, dtype=np.float64),
                raw_shapes=[str([1, cap, 768])] * len(sample.texts),
                sent_token_counts=np.full((len(sample.texts),), float(cap), dtype=np.float64),
                errors=[None] * len(sample.texts),
                load_seconds=0.01,
                processing_wall_seconds=0.1 * len(sample.texts),
                total_wall_seconds=0.11 * len(sample.texts),
                meta={"device": self.mode, "cap": int(cap), "batch_size": batch_size},
            )

    monkeypatch.setattr(
        benchmark_module,
        "_try_create_local_session",
        lambda model_name, device: (_FakeSession("gpu" if device == "cuda" else "cpu"), {"available": True}),
    )

    def _fake_hf_mode(*, sample, mode_name, cap=None, parallelism=1, **_kwargs):
        value = 3.0 if "parallel4" not in mode_name else 4.0
        vectors = np.vstack([np.full((1, 768), fill_value=value + idx, dtype=np.float32) for idx in range(len(sample.texts))])
        success_mask = np.ones((len(sample.texts),), dtype=bool)
        return benchmark_module._ModeRun(
            mode=str(mode_name),
            available=True,
            vectors=vectors,
            attempted_mask=np.ones((len(sample.texts),), dtype=bool),
            success_mask=success_mask,
            per_item_wall_seconds=np.full((len(sample.texts),), 0.3 if parallelism == 1 else 0.12, dtype=np.float64),
            raw_shapes=[str([cap or 128, 768])] * len(sample.texts),
            sent_token_counts=np.full((len(sample.texts),), float(cap or 128), dtype=np.float64),
            errors=[None] * len(sample.texts),
            load_seconds=0.0,
            processing_wall_seconds=(0.3 if parallelism == 1 else 0.12) * len(sample.texts),
            total_wall_seconds=(0.3 if parallelism == 1 else 0.12) * len(sample.texts),
            meta={"parallelism": int(parallelism), "cap": cap},
        )

    monkeypatch.setattr(benchmark_module, "_run_hf_mode", _fake_hf_mode)
    monkeypatch.setattr(
        benchmark_module,
        "_run_track_b_downstream",
        lambda **_kwargs: {
            "smoke": {
                "passed": True,
                "go_local": True,
                "go_hf": True,
                "mention_count_local": 8,
                "mention_count_hf": 8,
                "cluster_count_local": 4,
                "cluster_count_hf": 4,
                "changed_assignments": 0,
                "missing_mentions": 0,
            },
            "mini": {
                "wall_seconds": 12.5,
                "go": True,
                "mention_count": 8,
                "cluster_count": 4,
                "stage_metrics": {
                    "counts": {"ads_mentions": 8},
                    "runtime": {
                        "load_inputs": {"input_record_count": 2},
                        "pair_building": {"pairs_written": 6},
                    },
                },
            },
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "_compute_full_ads_scaling_stats",
        lambda **_kwargs: {"mentions_total": 80, "blocks_total": 10, "pair_upper_bound": 60},
    )

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-specter-benchmark",
            "--publications-path",
            str(tmp_path / "publications.parquet"),
            "--output-root",
            str(tmp_path / "benchmark"),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(tmp_path / "bundle"),
            "--no-progress",
        ]
    )
    payload = args.func(args)

    report_json = json.loads((tmp_path / "benchmark" / "specter_benchmark_report.json").read_text(encoding="utf-8"))
    report_md = (tmp_path / "benchmark" / "specter_benchmark_report.md").read_text(encoding="utf-8")

    assert payload["recommendation"] == report_json["decision"]["recommendation"]
    assert report_json["tracks"]["track_a"]["cap"] == 512
    assert report_json["tracks"]["track_b"]["cap"] == 256
    assert report_json["tracks"]["track_b"]["downstream"]["smoke"]["passed"] is True
    assert report_json["extrapolation"]["track_b"]["cpu_infer_tail"]["chosen_method"] == "pair_scaled"
    assert "SPECTER Benchmark Report" in report_md
    assert "Track B Downstream" in report_md
