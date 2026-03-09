from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Iterator, TypeVar

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

T = TypeVar("T")
_ACTIVE_UI: ContextVar["CliUI | None"] = ContextVar("active_cli_ui", default=None)


@dataclass
class _StepState:
    index: int
    label: str
    active: bool = True


class CliUI:
    """Tiny terminal UX helper for consistent CLI status + progress output."""

    def __init__(self, total_steps: int, progress: bool = True) -> None:
        self.total_steps = int(max(1, total_steps))
        self.current_step = 0
        self._state: _StepState | None = None
        show_bar = bool(progress) and sys.stderr.isatty()
        if _tqdm is None:
            self._bar = _NullProgressBar()
        else:
            self._bar = _tqdm(
                total=self.total_steps,
                disable=not show_bar,
                dynamic_ncols=True,
                unit="step",
                leave=False,
            )
        self._active_token = _ACTIVE_UI.set(self)

    def _prefix(self) -> str:
        step_idx = self.current_step if self.current_step > 0 else 1
        return f"[{step_idx:02d}/{self.total_steps:02d}]"

    def _write_line(self, line: str) -> None:
        if not self._bar.disable and hasattr(self._bar, "write"):
            self._bar.write(line)
            return
        print(line, file=sys.stderr, flush=True)

    def _emit(self, level: str, message: str) -> None:
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
        self._emit("WARN", message)

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
        _ACTIVE_UI.reset(self._active_token)


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
    min_plain_interval: float = 30.0

    def __post_init__(self) -> None:
        self.total = int(max(0, self.total))
        self.enabled = bool(self.enabled)
        self._started_at = perf_counter()
        self._done = 0
        self._last_plain_emit = 0.0
        self._plain_started = False
        self._completion_emitted = False
        show_bar = self.enabled and sys.stderr.isatty()
        if _tqdm is None:
            self._bar = _NullProgressBar()
        else:
            self._bar = _tqdm(
                total=self.total,
                disable=not show_bar,
                desc=self.label,
                dynamic_ncols=True,
                unit=self.unit,
                leave=False,
            )

    def __enter__(self) -> _LoopProgress:
        if self.enabled and self._bar.disable:
            self._emit_plain_start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        try:
            if exc is None:
                self.close()
            else:
                self._bar.close()
        finally:
            return None

    def _emit_plain_start(self) -> None:
        if self._plain_started:
            return
        self._plain_started = True
        prefix = _active_prefix()
        _write_progress_line(f"{prefix}INFO {self.label}: total={self.total} unit={self.unit}")

    def _emit_plain_snapshot(self, *, force: bool = False) -> None:
        if not self.enabled or not self._bar.disable:
            return
        self._emit_plain_start()
        now = perf_counter()
        if not force and (now - self._last_plain_emit) < float(self.min_plain_interval) and self._done < self.total:
            return
        elapsed = max(0.0, now - self._started_at)
        rate = (self._done / elapsed) if elapsed > 0 else 0.0
        eta = None if rate <= 0 or self.total <= 0 else max(0.0, (self.total - self._done) / rate)
        pct = _format_progress_percent(self._done, self.total)
        prefix = _active_prefix()
        _write_progress_line(
            f"{prefix}INFO {self.label}: progress={pct} done={self._done}/{self.total} "
            f"rate={rate:.2f}{self.unit}/s eta={_format_duration(eta)}"
        )
        self._last_plain_emit = now
        self._completion_emitted = self._done >= self.total

    def update(self, n: int = 1) -> None:
        if not self.enabled:
            return
        step = int(max(0, n))
        self._done += step
        if not self._bar.disable:
            self._bar.update(step)
            return
        self._emit_plain_snapshot()

    def close(self) -> None:
        if self.enabled and self._bar.disable and not self._completion_emitted:
            self._emit_plain_snapshot(force=True)
        self._bar.close()


def iter_progress(
    iterable: Iterable[T],
    *,
    total: int,
    label: str,
    enabled: bool,
    unit: str = "it",
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
    min_plain_interval: float = 30.0,
) -> Iterator[_LoopProgress]:
    tracker = _LoopProgress(
        label=label,
        total=total,
        enabled=enabled,
        unit=unit,
        min_plain_interval=min_plain_interval,
    )
    with tracker:
        yield tracker
