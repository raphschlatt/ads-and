from author_name_disambiguation.common import cli_ui


class _FakeBar:
    def __init__(self, *args, disable=False, **kwargs):
        self.disable = bool(disable)
        self.lines = []
        self.updates = 0

    def set_description_str(self, _desc: str) -> None:
        return None

    def update(self, n: int = 1) -> None:
        self.updates += int(n)

    def close(self) -> None:
        return None

    def write(self, line: str) -> None:
        self.lines.append(str(line))


class _FakeStderr:
    def isatty(self) -> bool:
        return True


class _FakeNonTtyStderr:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> None:
        self.lines.append(str(text))

    def flush(self) -> None:
        return None


def test_cli_ui_emits_start_line_when_progress_bar_active(monkeypatch):
    monkeypatch.setattr(cli_ui, "_tqdm", lambda *args, **kwargs: _FakeBar(*args, **kwargs))
    monkeypatch.setattr(cli_ui.sys, "stderr", _FakeStderr())

    ui = cli_ui.CliUI(total_steps=2, progress=True)
    ui.start("Load things")
    ui.done("Loaded.")
    ui.close()

    assert any(" START " in line for line in ui._bar.lines)
    assert any(" DONE " in line for line in ui._bar.lines)


def test_cli_ui_prints_start_line_when_progress_bar_disabled(capsys):
    ui = cli_ui.CliUI(total_steps=1, progress=False)
    ui.start("Prepare")
    ui.done("Prepared.")
    ui.close()

    err = capsys.readouterr().err
    assert "START Prepare" in err
    assert "DONE Prepared." in err


def test_iter_progress_plain_mode_writes_plain_lines(monkeypatch, capsys):
    fake_stderr = _FakeNonTtyStderr()
    monkeypatch.setattr(cli_ui.sys, "stderr", fake_stderr)

    list(
        cli_ui.iter_progress(
            range(3),
            total=3,
            label="SPECTER batches",
            enabled=True,
            unit="batch",
            min_plain_interval=0.0,
        )
    )

    err = "".join(fake_stderr.lines)
    assert "SPECTER batches: total=3 unit=batch" in err
    assert "progress=100% done=3/3" in err
    assert "\r" not in err
