import pytest

from author_name_disambiguation import cli


def test_build_parser_exposes_workspace_commands():
    parser = cli.build_parser()
    commands = set(parser._subparsers._group_actions[0].choices.keys())
    assert commands == {
        "infer",
        "quality-lspo",
        "train-lspo",
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
    assert args.progress_style == "compact"
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
    assert args.progress_style == "compact"
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
        ]
    )

    assert args.command == "run-infer-sources"
    assert args.publications_path == "publications.parquet"
    assert args.references_path is None
    assert args.output_root == "out"
    assert args.dataset_id == "my_ads_2026"
    assert args.model_bundle is None
    assert args.scratch_dir is None
    assert args.cluster_config is None
    assert args.gates_config is None
    assert args.infer_stage == "full"
    assert args.runtime_mode is None
    assert args.precision_mode == "fp32"
    assert args.cluster_backend is None
    assert args.uid_scope == "dataset"
    assert args.uid_namespace is None
    assert args.progress is True
    assert args.progress_style == "compact"
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
            "--scratch-dir",
            "/tmp/scratch",
            "--infer-stage",
            "incremental",
            "--cluster-config",
            "cfg/cluster.yaml",
            "--gates-config",
            "cfg/gates.yaml",
            "--no-progress",
            "--verbose-libs",
            "--runtime-mode",
            "cpu",
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
    assert args.scratch_dir == "/tmp/scratch"
    assert args.infer_stage == "incremental"
    assert args.cluster_config == "cfg/cluster.yaml"
    assert args.gates_config == "cfg/gates.yaml"
    assert args.progress is False
    assert args.progress_style == "compact"
    assert args.quiet_libs is False
    assert args.runtime_mode == "cpu"
    assert args.precision_mode == "amp_bf16"
    assert args.cluster_backend == "sklearn_cpu"
    assert args.uid_scope == "registry"
    assert args.uid_namespace == "stable_ads"


def test_infer_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            "out",
        ]
    )

    assert args.command == "infer"
    assert args.publications_path == "publications.parquet"
    assert args.references_path is None
    assert args.output_dir == "out"
    assert args.dataset_id is None
    assert args.model_bundle is None
    assert args.infer_stage == "full"
    assert args.runtime == "auto"
    assert args.force is False
    assert args.progress is True
    assert args.progress_style == "compact"
    assert args.quiet_libs is True
    assert args.json_output is False
    assert args.func is cli.cmd_infer


def test_quality_lspo_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["quality-lspo"])

    assert args.command == "quality-lspo"
    assert args.model_run_id is None
    assert args.model_bundle is None
    assert args.data_root == "data"
    assert args.artifacts_root == "artifacts"
    assert args.raw_lspo_parquet == "data/raw/lspo/LSPO_v1.parquet"
    assert args.raw_lspo_h5 is None
    assert args.device == "auto"
    assert args.precision_mode == "fp32"
    assert args.score_batch_size == 8192
    assert args.report_tag is None
    assert args.force is False
    assert args.progress is True
    assert args.progress_style == "compact"
    assert args.quiet_libs is True
    assert args.json_output is False
    assert args.func is cli.cmd_quality_lspo


def test_train_lspo_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["train-lspo"])

    assert args.command == "train-lspo"
    assert args.run_stage == "full"
    assert args.data_root == "data"
    assert args.artifacts_root == "artifacts"
    assert args.raw_lspo_parquet == "data/raw/lspo/LSPO_v1.parquet"
    assert args.raw_lspo_h5 is None
    assert args.run_id is None
    assert args.device == "auto"
    assert args.precision_mode is None
    assert args.force is False
    assert args.progress is True
    assert args.progress_style == "compact"
    assert args.quiet_libs is True
    assert args.json_output is False
    assert args.func is cli.cmd_train_lspo


def test_simple_commands_accept_verbose_progress_and_json():
    parser = cli.build_parser()

    infer_args = parser.parse_args(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            "out",
            "--verbose-progress",
            "--json",
        ]
    )
    assert infer_args.progress_style == "verbose"
    assert infer_args.json_output is True

    quality_args = parser.parse_args(["quality-lspo", "--verbose-progress", "--json"])
    assert quality_args.progress_style == "verbose"
    assert quality_args.json_output is True

    train_args = parser.parse_args(["train-lspo", "--verbose-progress", "--json"])
    assert train_args.progress_style == "verbose"
    assert train_args.json_output is True


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
