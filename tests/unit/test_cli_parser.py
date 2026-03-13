import pytest

from author_name_disambiguation import cli


def test_build_parser_exposes_only_public_commands():
    parser = cli.build_parser()
    commands = set(parser._subparsers._group_actions[0].choices.keys())
    assert commands == {
        "run-train-stage",
        "run-infer-sources",
        "compare-infer-baseline",
        "run-cluster-test-report",
        "export-model-bundle",
    }


def test_run_train_stage_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-train-stage",
            "--run-stage",
            "smoke",
            "--data-root",
            "/tmp/data",
            "--artifacts-root",
            "/tmp/artifacts",
            "--raw-lspo-parquet",
            "/tmp/data/raw/lspo/mock.parquet",
        ]
    )

    assert args.command == "run-train-stage"
    assert args.run_stage == "smoke"
    assert args.data_root == "/tmp/data"
    assert args.artifacts_root == "/tmp/artifacts"
    assert args.raw_lspo_parquet == "/tmp/data/raw/lspo/mock.parquet"
    assert args.raw_lspo_h5 is None
    assert args.progress is True
    assert args.quiet_libs is True
    assert args.seeds is None
    assert args.device == "auto"
    assert args.precision_mode is None
    assert args.score_batch_size == 8192
    assert args.func is cli.cmd_run_train_stage


def test_run_train_stage_parser_overrides():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-train-stage",
            "--run-stage",
            "mini",
            "--data-root",
            "/tmp/data",
            "--artifacts-root",
            "/tmp/artifacts",
            "--raw-lspo-parquet",
            "/tmp/data/raw/lspo/mock.parquet",
            "--raw-lspo-h5",
            "/tmp/data/raw/lspo/mock.h5",
            "--run-config",
            "cfg/run.yaml",
            "--model-config",
            "cfg/model.yaml",
            "--cluster-config",
            "cfg/cluster.yaml",
            "--gates-config",
            "cfg/gates.yaml",
            "--run-id",
            "mini_run",
            "--precision-mode",
            "amp_bf16",
            "--seeds",
            "3",
            "5",
            "--baseline-run-id",
            "baseline_1",
            "--force",
            "--no-progress",
            "--verbose-libs",
        ]
    )

    assert args.run_stage == "mini"
    assert args.raw_lspo_h5 == "/tmp/data/raw/lspo/mock.h5"
    assert args.run_config == "cfg/run.yaml"
    assert args.model_config == "cfg/model.yaml"
    assert args.cluster_config == "cfg/cluster.yaml"
    assert args.gates_config == "cfg/gates.yaml"
    assert args.run_id == "mini_run"
    assert args.precision_mode == "amp_bf16"
    assert args.seeds == [3, 5]
    assert args.baseline_run_id == "baseline_1"
    assert args.force is True
    assert args.progress is False
    assert args.quiet_libs is False


def test_run_infer_sources_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            "publications.parquet",
            "--output-root",
            "out",
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            "/tmp/bundle",
        ]
    )

    assert args.command == "run-infer-sources"
    assert args.publications_path == "publications.parquet"
    assert args.references_path is None
    assert args.output_root == "out"
    assert args.dataset_id == "my_ads_2026"
    assert args.model_bundle == "/tmp/bundle"
    assert args.cluster_config is None
    assert args.gates_config is None
    assert args.infer_stage == "full"
    assert args.device == "auto"
    assert args.precision_mode == "fp32"
    assert args.cluster_backend is None
    assert args.uid_scope == "dataset"
    assert args.uid_namespace is None
    assert args.progress is True
    assert args.quiet_libs is True
    assert args.func is cli.cmd_run_infer_sources


def test_run_infer_sources_parser_accepts_overrides():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            "publications.parquet",
            "--references-path",
            "references.parquet",
            "--output-root",
            "out",
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            "/tmp/bundle",
            "--infer-stage",
            "mini",
            "--cluster-config",
            "cfg/cluster.yaml",
            "--gates-config",
            "cfg/gates.yaml",
            "--no-progress",
            "--verbose-libs",
            "--precision-mode",
            "amp_bf16",
            "--cluster-backend",
            "sklearn_cpu",
            "--uid-scope",
            "registry",
            "--uid-namespace",
            "stable_ads",
        ]
    )

    assert args.references_path == "references.parquet"
    assert args.infer_stage == "mini"
    assert args.cluster_config == "cfg/cluster.yaml"
    assert args.gates_config == "cfg/gates.yaml"
    assert args.progress is False
    assert args.quiet_libs is False
    assert args.precision_mode == "amp_bf16"
    assert args.cluster_backend == "sklearn_cpu"
    assert args.uid_scope == "registry"
    assert args.uid_namespace == "stable_ads"


def test_export_model_bundle_parser():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "export-model-bundle",
            "--model-run-id",
            "full_2026abc",
            "--artifacts-root",
            "/tmp/artifacts",
            "--output-dir",
            "/tmp/out",
        ]
    )
    assert args.command == "export-model-bundle"
    assert args.model_run_id == "full_2026abc"
    assert args.artifacts_root == "/tmp/artifacts"
    assert args.output_dir == "/tmp/out"
    assert args.func is cli.cmd_export_model_bundle


def test_run_cluster_test_report_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-cluster-test-report",
            "--model-run-id",
            "full_2026abc",
            "--data-root",
            "/tmp/data",
            "--artifacts-root",
            "/tmp/artifacts",
            "--raw-lspo-parquet",
            "/tmp/data/raw/lspo/mock.parquet",
        ]
    )
    assert args.command == "run-cluster-test-report"
    assert args.model_run_id == "full_2026abc"
    assert args.data_root == "/tmp/data"
    assert args.artifacts_root == "/tmp/artifacts"
    assert args.raw_lspo_parquet == "/tmp/data/raw/lspo/mock.parquet"
    assert args.raw_lspo_h5 is None
    assert args.device == "auto"
    assert args.precision_mode == "fp32"
    assert args.score_batch_size == 8192
    assert args.cluster_config_override is None
    assert args.report_tag is None
    assert args.allow_legacy_lspo_compat is False
    assert args.force is False
    assert args.progress is True
    assert args.quiet_libs is True
    assert args.func is cli.cmd_run_cluster_test_report


def test_run_cluster_test_report_parser_overrides():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-cluster-test-report",
            "--model-run-id",
            "full_2026abc",
            "--data-root",
            "/tmp/data",
            "--artifacts-root",
            "/tmp/artifacts",
            "--raw-lspo-parquet",
            "/tmp/data/raw/lspo/mock.parquet",
            "--raw-lspo-h5",
            "/tmp/data/raw/lspo/mock.h5",
            "--device",
            "cpu",
            "--precision-mode",
            "amp_bf16",
            "--score-batch-size",
            "4096",
            "--cluster-config-override",
            "cfg/cluster-override.yaml",
            "--report-tag",
            "tightened",
            "--allow-legacy-lspo-compat",
            "--force",
            "--no-progress",
            "--verbose-libs",
        ]
    )
    assert args.raw_lspo_h5 == "/tmp/data/raw/lspo/mock.h5"
    assert args.device == "cpu"
    assert args.precision_mode == "amp_bf16"
    assert args.score_batch_size == 4096
    assert args.cluster_config_override == "cfg/cluster-override.yaml"
    assert args.report_tag == "tightened"
    assert args.allow_legacy_lspo_compat is True
    assert args.force is True
    assert args.progress is False
    assert args.quiet_libs is False


def test_removed_commands_are_not_exposed():
    parser = cli.build_parser()
    for argv in [
        ["run-stage", "--run-stage", "smoke"],
        ["cache", "doctor"],
        ["prepare-lspo"],
        ["prepare-ads"],
        ["embeddings"],
        ["pairs"],
        ["train"],
        ["score"],
        ["cluster"],
        ["export"],
        ["report"],
    ]:
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
