import pytest

from author_name_disambiguation import cli


def test_run_train_stage_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["run-train-stage", "--run-stage", "smoke"])

    assert args.command == "run-train-stage"
    assert args.run_stage == "smoke"
    assert args.progress is True
    assert args.quiet_libs is True
    assert args.seeds is None
    assert args.device == "auto"
    assert args.precision_mode is None
    assert args.score_batch_size == 8192
    assert args.export_model_bundle is True
    assert args.func is cli.cmd_run_train_stage


def test_run_stage_alias_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["run-stage", "--run-stage", "smoke"])

    assert args.command == "run-stage"
    assert args.run_stage == "smoke"
    assert args.prefer_precomputed_ads is True
    assert args.export_model_bundle is True
    assert args.func is cli.cmd_run_stage


def test_run_stage_alias_parser_boolean_overrides():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-stage",
            "--run-stage",
            "mini",
            "--no-prefer-precomputed-ads",
            "--no-progress",
            "--verbose-libs",
            "--no-export-model-bundle",
            "--seeds",
            "3",
            "5",
        ]
    )

    assert args.run_stage == "mini"
    assert args.prefer_precomputed_ads is False
    assert args.progress is False
    assert args.quiet_libs is False
    assert args.export_model_bundle is False
    assert args.seeds == [3, 5]


def test_embeddings_parser_quiet_libs_flags():
    parser = cli.build_parser()
    args_default = parser.parse_args(
        [
            "embeddings",
            "--mentions",
            "x.parquet",
            "--chars-out",
            "chars.npy",
            "--text-out",
            "text.npy",
        ]
    )
    assert args_default.quiet_libs is True

    args_verbose = parser.parse_args(
        [
            "embeddings",
            "--mentions",
            "x.parquet",
            "--chars-out",
            "chars.npy",
            "--text-out",
            "text.npy",
            "--verbose-libs",
        ]
    )
    assert args_verbose.quiet_libs is False


def test_score_parser_precision_default():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "score",
            "--mentions",
            "m.parquet",
            "--pairs",
            "p.parquet",
            "--chars",
            "c.npy",
            "--text",
            "t.npy",
            "--checkpoint",
            "ckpt.pt",
            "--output",
            "scores.parquet",
        ]
    )
    assert args.precision_mode == "fp32"


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
    assert args.func is cli.cmd_run_infer_sources


def test_run_infer_sources_parser_accepts_references_and_overrides():
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
    assert args.model_bundle == "/tmp/bundle"
    assert args.infer_stage == "mini"
    assert args.cluster_config == "cfg/cluster.yaml"
    assert args.gates_config == "cfg/gates.yaml"
    assert args.progress is False
    assert args.precision_mode == "amp_bf16"
    assert args.cluster_backend == "sklearn_cpu"
    assert args.uid_scope == "registry"
    assert args.uid_namespace == "stable_ads"


def test_run_infer_sources_parser_requires_required_fields():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-infer-sources",
                "--dataset-id",
                "my_ads_2026",
                "--publications-path",
                "publications.parquet",
                "--output-root",
                "out",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-infer-sources",
                "--publications-path",
                "publications.parquet",
                "--output-root",
                "out",
                "--model-bundle",
                "/tmp/bundle",
            ]
        )


def test_export_model_bundle_parser():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "export-model-bundle",
            "--model-run-id",
            "full_2026abc",
            "--output-dir",
            "/tmp/out",
        ]
    )
    assert args.command == "export-model-bundle"
    assert args.model_run_id == "full_2026abc"
    assert args.output_dir == "/tmp/out"
    assert args.func is cli.cmd_export_model_bundle


def test_run_cluster_test_report_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-cluster-test-report",
            "--model-run-id",
            "full_2026abc",
        ]
    )
    assert args.command == "run-cluster-test-report"
    assert args.model_run_id == "full_2026abc"
    assert args.paths_config == "configs/paths.local.yaml"
    assert args.device == "auto"
    assert args.precision_mode == "fp32"
    assert args.score_batch_size == 8192
    assert args.cluster_config_override is None
    assert args.report_tag is None
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
            "--paths-config",
            "cfg/paths.yaml",
            "--device",
            "cpu",
            "--precision-mode",
            "amp_bf16",
            "--score-batch-size",
            "4096",
            "--cluster-config-override",
            "cfg/cluster.yaml",
            "--report-tag",
            "epsbkt_v1",
            "--force",
            "--no-progress",
            "--verbose-libs",
        ]
    )
    assert args.paths_config == "cfg/paths.yaml"
    assert args.device == "cpu"
    assert args.precision_mode == "amp_bf16"
    assert args.score_batch_size == 4096
    assert args.cluster_config_override == "cfg/cluster.yaml"
    assert args.report_tag == "epsbkt_v1"
    assert args.force is True
    assert args.progress is False
    assert args.quiet_libs is False
