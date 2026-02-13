from src import cli


def test_run_stage_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["run-stage", "--run-stage", "smoke"])

    assert args.command == "run-stage"
    assert args.run_stage == "smoke"
    assert args.prefer_precomputed_ads is True
    assert args.progress is True
    assert args.device == "auto"
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
        ]
    )

    assert args.run_stage == "mini"
    assert args.prefer_precomputed_ads is False
    assert args.progress is False
