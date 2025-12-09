from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class ProgressSnapshot:
    step: Optional[int] = None
    total: Optional[int] = None
    percent: Optional[float] = None
    phase: Optional[str] = None
    message: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Lightweight progress helper shared between backend and training loops."""

    def __init__(self, publisher: Optional[Callable[[ProgressSnapshot], None]] = None,
                 throttle_seconds: float = 1.0):
        self.publisher = publisher
        self.throttle_seconds = throttle_seconds
        self._last_emit = 0.0
        self.snapshot = ProgressSnapshot()

    def update(self,
               *,
               step: Optional[int] = None,
               total: Optional[int] = None,
               phase: Optional[str] = None,
               message: Optional[str] = None,
               info: Optional[Dict[str, Any]] = None,
               force: bool = False) -> None:
        if step is not None:
            self.snapshot.step = step
        if total is not None:
            self.snapshot.total = total
        if phase is not None:
            self.snapshot.phase = phase
        if message is not None:
            self.snapshot.message = message
        if info is not None:
            self.snapshot.info = info
        if self.snapshot.step is not None and self.snapshot.total:
            self.snapshot.percent = (self.snapshot.step / self.snapshot.total) * 100.0
        now = time.time()
        if self.publisher is None:
            return
        if not force and (now - self._last_emit) < self.throttle_seconds:
            return
        self._last_emit = now
        self.publisher(self.snapshot)

    def complete(self, message: Optional[str] = None) -> None:
        if self.snapshot.total is not None:
            self.snapshot.step = self.snapshot.total
        self.snapshot.percent = 100.0
        self.snapshot.phase = self.snapshot.phase or "completed"
        if message:
            self.snapshot.message = message
        self.update(force=True)

