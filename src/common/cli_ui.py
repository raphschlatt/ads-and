from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


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

    def _prefix(self) -> str:
        step_idx = self.current_step if self.current_step > 0 else 1
        return f"[{step_idx:02d}/{self.total_steps:02d}]"

    def _emit(self, level: str, message: str) -> None:
        print(f"{self._prefix()} {level} {message}")

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


class _NullProgressBar:
    disable = True

    def set_description_str(self, _desc: str) -> None:
        return None

    def update(self, _n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None
