from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Iterator, TypeVar

from author_name_disambiguation.progress import ProgressEvent, emit_stage_progress, get_active_progress_reporter

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

T = TypeVar("T")
_ACTIVE_UI: ContextVar["CliUI | None"] = ContextVar("active_cli_ui", default=None)
_ACTIVE_PROGRESS_BAR: ContextVar[object | None] = ContextVar("active_progress_bar", default=None)
_NESTED_PROGRESS_ENABLED: ContextVar[bool] = ContextVar("nested_progress_enabled", default=True)


def _is_notebook_environment() -> bool:
    try:  # pragma: no cover - environment dependent
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _supports_live_progress() -> bool:
    return bool(sys.stderr.isatty() or _is_notebook_environment())


@dataclass
class _StepState:
    index: int
    label: str
    active: bool = True


class CliUI:
    """Tiny terminal UX helper for consistent CLI status + progress output."""

    def __init__(self, total_steps: int, progress: bool = True, progress_style: str = "compact") -> None:
        self.total_steps = int(max(1, total_steps))
        self.current_step = 0
        self._state: _StepState | None = None
        self._warnings: list[str] = []
        self._warning_set: set[str] = set()
        self._notebook_environment = _is_notebook_environment()
        normalized_style = str(progress_style or "compact").strip().lower() or "compact"
        if normalized_style not in {"compact", "verbose"}:
            normalized_style = "compact"
        self.progress_style = normalized_style
        self._mode = "silent" if not bool(progress) else self.progress_style
        show_bar = bool(progress) and bool(sys.stderr.isatty() or self._notebook_environment)
        leave_bar = bool(self._mode == "compact" or self._notebook_environment)
        if _tqdm is None or not show_bar:
            self._bar = _NullProgressBar()
        else:
            self._bar = _tqdm(
                total=self.total_steps,
                disable=False,
                dynamic_ncols=True,
                unit="step",
                leave=leave_bar,
            )
        self._active_token = _ACTIVE_UI.set(self)
        self._nested_progress_token = _NESTED_PROGRESS_ENABLED.set(self._mode == "verbose")

    def _prefix(self) -> str:
        step_idx = self.current_step if self.current_step > 0 else 1
        return f"[{step_idx:02d}/{self.total_steps:02d}]"

    def _write_line(self, line: str) -> None:
        active_progress_bar = _ACTIVE_PROGRESS_BAR.get()
        if active_progress_bar is not None and hasattr(active_progress_bar, "write"):
            active_progress_bar.write(line)
            return
        if not self._bar.disable and hasattr(self._bar, "write"):
            self._bar.write(line)
            return
        print(line, file=sys.stderr, flush=True)

    def _emit(self, level: str, message: str) -> None:
        if self._mode == "silent":
            return
        if self._mode == "compact" and level != "FAIL":
            return
        line = f"{self._prefix()} {level} {message}"
        self._write_line(line)

    def start(self, label: str) -> None:
        if self._state and self._state.active:
            self.done("Completed.")
        self.current_step += 1
        self._state = _StepState(index=self.current_step, label=str(label))
        if not self._bar.disable:
            self._bar.set_description_str(f"{self.current_step:02d}/{self.total_steps:02d} {label}")
        self._emit("START", label)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        text = str(message)
        if self._mode == "silent":
            return
        if self._mode == "compact":
            if text not in self._warning_set:
                self._warning_set.add(text)
                self._warnings.append(text)
            return
        self._emit("WARN", text)

    def done(self, message: str = "Done.") -> None:
        if self._state and self._state.active:
            self._state.active = False
            self._bar.update(1)
        self._emit("DONE", message)

    def skip(self, message: str = "Reused existing artifacts.") -> None:
        if self._state and self._state.active:
            self._state.active = False
            self._bar.update(1)
        self._emit("SKIP", message)

    def fail(self, message: str = "Failed.") -> None:
        if self._state and self._state.active:
            self._state.active = False
            self._bar.update(1)
        self._emit("FAIL", message)

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        self.start(label)
        t0 = perf_counter()
        try:
            yield
        except Exception as exc:
            self.fail(f"{label} ({exc})")
            raise
        else:
            elapsed = perf_counter() - t0
            self.done(f"Completed in {elapsed:.2f}s.")

    def close(self) -> None:
        self._bar.close()
        if self._mode == "compact" and self._warnings:
            print("Warnings: " + " | ".join(self._warnings), file=sys.stderr, flush=True)
        _NESTED_PROGRESS_ENABLED.reset(self._nested_progress_token)
        _ACTIVE_UI.reset(self._active_token)

    def current_stage_label(self) -> str | None:
        if self._state is None or not self._state.active:
            return None
        return str(self._state.label)


class _NullProgressBar:
    disable = True

    def set_description_str(self, _desc: str) -> None:
        return None

    def update(self, _n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def get_active_ui() -> CliUI | None:
    return _ACTIVE_UI.get()


def nested_progress_enabled() -> bool:
    return bool(_NESTED_PROGRESS_ENABLED.get())


class CliProgressHandler:
    def __init__(self, ui: CliUI) -> None:
        self._ui = ui

    def __call__(self, event: ProgressEvent) -> None:
        kind = str(event.kind)
        if kind == "stage_start":
            self._ui.start(event.stage_label or "")
            return
        if kind == "stage_info":
            if event.message:
                self._ui.info(str(event.message))
            return
        if kind == "stage_warning":
            if event.message:
                self._ui.warn(str(event.message))
            return
        if kind == "stage_done":
            message = str(event.message or "Done.")
            status = str((event.payload or {}).get("status", "")).strip().lower()
            if status == "skipped":
                self._ui.skip(message)
            else:
                self._ui.done(message)
            return
        if kind == "run_failed":
            if event.message:
                self._ui.fail(str(event.message))
            return
        if kind == "run_done":
            if event.message:
                self._ui.info(str(event.message))
            return
        if kind == "stage_progress":
            return


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_progress_percent(done: int, total: int) -> str:
    if total <= 0:
        return "100%"
    if done <= 0:
        return "0%"

    pct = (float(done) * 100.0) / float(total)
    if pct < 1.0:
        return "<1%"
    if pct < 10.0:
        return f"{pct:.1f}%"
    return f"{int(pct)}%"


def _active_prefix() -> str:
    ui = get_active_ui()
    if ui is None:
        return ""
    return f"{ui._prefix()} "


def _write_progress_line(line: str) -> None:
    ui = get_active_ui()
    if ui is not None:
        ui._write_line(line)
        return
    print(line, file=sys.stderr, flush=True)


@dataclass
class _LoopProgress:
    label: str
    total: int
    enabled: bool
    unit: str
    compact_label: str | None = None
    compact_visible: bool = True
    emit_events: bool = True
    min_plain_interval: float = 30.0

    def __post_init__(self) -> None:
        self.total = int(max(0, self.total))
        self.enabled = bool(self.enabled) and nested_progress_enabled()
        self._ui = get_active_ui()
        reporter = get_active_progress_reporter()
        self._external_only = bool(
            reporter is not None and reporter.has_handler() and not isinstance(reporter.handler, CliProgressHandler)
        )
        self._compact_mode = bool(self._ui is not None and self._ui.progress_style == "compact")
        self._compact_hidden = bool(self.enabled and self._compact_mode and not bool(self.compact_visible))
        self._suppress_output = bool(self._external_only or self._compact_hidden)
        self._started_at = perf_counter()
        self._done = 0
        self._last_plain_emit = 0.0
        self._plain_started = False
        self._completion_emitted = False
        self._display_label = str(self.label)
        self._plain_prefix = _active_prefix()
        self._active_bar_token = None

        if self.enabled and self._compact_mode and not self._suppress_output and self._ui is not None:
            compact_display = str(self.compact_label or self._ui.current_stage_label() or self.label)
            self._display_label = f"{self._ui._prefix()} {compact_display}"
            self._plain_prefix = ""

        show_bar = self.enabled and not self._suppress_output and _supports_live_progress()
        leave_bar = bool((self._compact_mode or _is_notebook_environment()) and not self._suppress_output)
        if _tqdm is None or not self.enabled or self._suppress_output:
            self._bar = _NullProgressBar()
        else:
            self._bar = _tqdm(
                total=self.total,
                disable=not show_bar,
                desc=self._display_label,
                dynamic_ncols=True,
                unit=self.unit,
                leave=leave_bar,
            )

    def __enter__(self) -> _LoopProgress:
        if self.enabled and not self._suppress_output and not self._bar.disable:
            self._active_bar_token = _ACTIVE_PROGRESS_BAR.set(self._bar)
        elif self.enabled and not self._suppress_output and self._bar.disable:
            self._emit_plain_start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        try:
            if exc is None:
                self.close()
            else:
                self._bar.close()
        finally:
            if self._active_bar_token is not None:
                _ACTIVE_PROGRESS_BAR.reset(self._active_bar_token)
                self._active_bar_token = None
            return None

    def _emit_plain_start(self) -> None:
        if self._plain_started:
            return
        self._plain_started = True
        _write_progress_line(f"{self._plain_prefix}INFO {self._display_label}: total={self.total} unit={self.unit}")

    def _emit_plain_snapshot(self, *, force: bool = False) -> None:
        if not self.enabled or self._suppress_output or not self._bar.disable:
            return
        self._emit_plain_start()
        now = perf_counter()
        if not force and (now - self._last_plain_emit) < float(self.min_plain_interval) and self._done < self.total:
            return
        elapsed = max(0.0, now - self._started_at)
        rate = (self._done / elapsed) if elapsed > 0 else 0.0
        eta = None if rate <= 0 or self.total <= 0 else max(0.0, (self.total - self._done) / rate)
        pct = _format_progress_percent(self._done, self.total)
        _write_progress_line(
            f"{self._plain_prefix}INFO {self._display_label}: progress={pct} done={self._done}/{self.total} "
            f"rate={rate:.2f}{self.unit}/s eta={_format_duration(eta)}"
        )
        self._last_plain_emit = now
        self._completion_emitted = self._done >= self.total

    def update(self, n: int = 1) -> None:
        if not self.enabled or self._compact_hidden:
            return
        step = int(max(0, n))
        self._done += step
        if bool(self.emit_events):
            emit_stage_progress(
                current=self._done,
                total=self.total,
                unit=self.unit,
                payload={"progress_label": str(self.label)},
            )
        if self._external_only:
            return
        if not self._bar.disable:
            self._bar.update(step)
            return
        self._emit_plain_snapshot()

    def close(self) -> None:
        if self.enabled and not self._suppress_output and self._bar.disable and not self._completion_emitted:
            self._emit_plain_snapshot(force=True)
        self._bar.close()


def iter_progress(
    iterable: Iterable[T],
    *,
    total: int,
    label: str,
    enabled: bool,
    unit: str = "it",
    compact_label: str | None = None,
    compact_visible: bool = True,
    emit_events: bool = True,
    min_plain_interval: float = 30.0,
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return

    tracker = _LoopProgress(
        label=label,
        total=total,
        enabled=enabled,
        unit=unit,
        compact_label=compact_label,
        compact_visible=compact_visible,
        emit_events=emit_events,
        min_plain_interval=min_plain_interval,
    )
    with tracker:
        for item in iterable:
            yield item
            tracker.update(1)


@contextmanager
def loop_progress(
    *,
    total: int,
    label: str,
    enabled: bool,
    unit: str = "it",
    compact_label: str | None = None,
    compact_visible: bool = True,
    emit_events: bool = True,
    min_plain_interval: float = 30.0,
) -> Iterator[_LoopProgress]:
    tracker = _LoopProgress(
        label=label,
        total=total,
        enabled=enabled,
        unit=unit,
        compact_label=compact_label,
        compact_visible=compact_visible,
        emit_events=emit_events,
        min_plain_interval=min_plain_interval,
    )
    with tracker:
        yield tracker
