from src import cli


def test_run_stage_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["run-stage", "--run-stage", "smoke"])

    assert args.command == "run-stage"
    assert args.run_stage == "smoke"
    assert args.prefer_precomputed_ads is True
    assert args.progress is True
    assert args.quiet_libs is True
    assert args.seeds is None
    assert args.device == "auto"
    assert args.precision_mode is None
    assert args.score_batch_size == 8192
    assert args.func is cli.cmd_run_stage


def test_run_stage_parser_boolean_overrides():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-stage",
            "--run-stage",
            "mini",
            "--no-prefer-precomputed-ads",
            "--no-progress",
            "--verbose-libs",
            "--seeds",
            "3",
            "5",
        ]
    )

    assert args.run_stage == "mini"
    assert args.prefer_precomputed_ads is False
    assert args.progress is False
    assert args.quiet_libs is False
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
