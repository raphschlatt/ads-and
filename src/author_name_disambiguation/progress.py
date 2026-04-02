from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal


ProgressKind = Literal[
    "stage_start",
    "stage_info",
    "stage_progress",
    "stage_warning",
    "stage_done",
    "run_done",
    "run_failed",
]
ProgressHandler = Callable[["ProgressEvent"], None]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    kind: ProgressKind
    stage_index: int | None = None
    stage_total: int | None = None
    stage_key: str | None = None
    stage_label: str | None = None
    message: str | None = None
    current: int | None = None
    total: int | None = None
    unit: str | None = None
    elapsed_seconds: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _StageState:
    stage_index: int
    stage_total: int
    stage_key: str
    stage_label: str


_ACTIVE_PROGRESS_REPORTER: ContextVar["ProgressReporter | None"] = ContextVar("active_progress_reporter", default=None)


class ProgressReporter:
    def __init__(self, handler: ProgressHandler | None = None) -> None:
        self._handler = handler
        self._stage_state: _StageState | None = None

    @property
    def handler(self) -> ProgressHandler | None:
        return self._handler

    def has_handler(self) -> bool:
        return self._handler is not None

    def _emit(self, event: ProgressEvent) -> None:
        if self._handler is None:
            return
        self._handler(event)

    def current_stage(self) -> _StageState | None:
        return self._stage_state

    def start_stage(self, *, stage_index: int, stage_total: int, stage_key: str, stage_label: str) -> None:
        self._stage_state = _StageState(
            stage_index=int(stage_index),
            stage_total=int(stage_total),
            stage_key=str(stage_key),
            stage_label=str(stage_label),
        )
        self._emit(
            ProgressEvent(
                kind="stage_start",
                stage_index=self._stage_state.stage_index,
                stage_total=self._stage_state.stage_total,
                stage_key=self._stage_state.stage_key,
                stage_label=self._stage_state.stage_label,
                payload={},
            )
        )

    def info(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        stage = self._stage_state
        self._emit(
            ProgressEvent(
                kind="stage_info",
                stage_index=None if stage is None else stage.stage_index,
                stage_total=None if stage is None else stage.stage_total,
                stage_key=None if stage is None else stage.stage_key,
                stage_label=None if stage is None else stage.stage_label,
                message=str(message),
                payload=dict(payload or {}),
            )
        )

    def warn(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        stage = self._stage_state
        self._emit(
            ProgressEvent(
                kind="stage_warning",
                stage_index=None if stage is None else stage.stage_index,
                stage_total=None if stage is None else stage.stage_total,
                stage_key=None if stage is None else stage.stage_key,
                stage_label=None if stage is None else stage.stage_label,
                message=str(message),
                payload=dict(payload or {}),
            )
        )

    def progress(
        self,
        *,
        current: int,
        total: int,
        unit: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        stage = self._stage_state
        if stage is None:
            return
        self._emit(
            ProgressEvent(
                kind="stage_progress",
                stage_index=stage.stage_index,
                stage_total=stage.stage_total,
                stage_key=stage.stage_key,
                stage_label=stage.stage_label,
                current=int(current),
                total=int(total),
                unit=str(unit),
                payload=dict(payload or {}),
            )
        )

    def done(
        self,
        message: str,
        *,
        elapsed_seconds: float | None = None,
        payload: dict[str, Any] | None = None,
        skipped: bool = False,
    ) -> None:
        stage = self._stage_state
        self._emit(
            ProgressEvent(
                kind="stage_done",
                stage_index=None if stage is None else stage.stage_index,
                stage_total=None if stage is None else stage.stage_total,
                stage_key=None if stage is None else stage.stage_key,
                stage_label=None if stage is None else stage.stage_label,
                message=str(message),
                elapsed_seconds=None if elapsed_seconds is None else float(elapsed_seconds),
                payload={**({"status": "skipped"} if skipped else {}), **dict(payload or {})},
            )
        )
        self._stage_state = None

    def run_done(self, *, payload: dict[str, Any] | None = None, message: str | None = None) -> None:
        self._emit(
            ProgressEvent(
                kind="run_done",
                message=None if message is None else str(message),
                payload=dict(payload or {}),
            )
        )

    def run_failed(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        stage = self._stage_state
        self._emit(
            ProgressEvent(
                kind="run_failed",
                stage_index=None if stage is None else stage.stage_index,
                stage_total=None if stage is None else stage.stage_total,
                stage_key=None if stage is None else stage.stage_key,
                stage_label=None if stage is None else stage.stage_label,
                message=str(message),
                payload=dict(payload or {}),
            )
        )


@contextmanager
def activate_progress_reporter(reporter: ProgressReporter | None) -> Iterator[ProgressReporter | None]:
    token = set_active_progress_reporter(reporter)
    try:
        yield reporter
    finally:
        reset_active_progress_reporter(token)


def set_active_progress_reporter(reporter: ProgressReporter | None):
    return _ACTIVE_PROGRESS_REPORTER.set(reporter)


def reset_active_progress_reporter(token) -> None:
    _ACTIVE_PROGRESS_REPORTER.reset(token)


def get_active_progress_reporter() -> ProgressReporter | None:
    return _ACTIVE_PROGRESS_REPORTER.get()


def emit_stage_progress(
    *,
    current: int,
    total: int,
    unit: str,
    payload: dict[str, Any] | None = None,
) -> None:
    reporter = get_active_progress_reporter()
    if reporter is None or not reporter.has_handler():
        return
    reporter.progress(
        current=int(current),
        total=int(total),
        unit=str(unit),
        payload=dict(payload or {}),
    )
