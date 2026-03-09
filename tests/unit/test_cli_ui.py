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


def test_cli_ui_avoids_start_line_when_progress_bar_active(monkeypatch):
    monkeypatch.setattr(cli_ui, "_tqdm", lambda *args, **kwargs: _FakeBar(*args, **kwargs))
    monkeypatch.setattr(cli_ui.sys, "stderr", _FakeStderr())

    ui = cli_ui.CliUI(total_steps=2, progress=True)
    ui.start("Load things")
    ui.done("Loaded.")

    assert not any(" START " in line for line in ui._bar.lines)
    assert any(" DONE " in line for line in ui._bar.lines)


def test_cli_ui_prints_start_line_when_progress_bar_disabled(capsys):
    ui = cli_ui.CliUI(total_steps=1, progress=False)
    ui.start("Prepare")
    ui.done("Prepared.")

    out = capsys.readouterr().out
    assert "START Prepare" in out
    assert "DONE Prepared." in out
