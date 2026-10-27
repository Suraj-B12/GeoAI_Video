"""Structured run logs (docs/HANDOFF_PLAN.md §9): one JSON object per line, beside the human log lines.

Each run of a command on a session appends to `<root>/logs/<UTC date>/<session>.jsonl`, where root is
the command's output folder. Every line carries the UTC time, the run id, the session, the command, a
stage and an event:
- `start` / `end` of each stage, `end` with the stage's wall time and the counts the stage reports;
- `error` with the exception's type, message and full stack trace, after which the error propagates;
- `info` for anything else worth keeping.
The operator UI and later analyses read these files instead of parsing free text. They are append-only:
reruns on the same day add lines with a new run id, so nothing earlier is lost. Times and the run id
make the logs differ from run to run; they are never part of a run's deterministic outputs.
"""
from __future__ import annotations

import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .util import to_json


def _utc_iso(t: float) -> str:
    d = datetime.fromtimestamp(t, tz=timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


class RunLog:
    """The structured log of one command run on one session (see the module docstring)."""

    def __init__(self, root: Path, session: str, command: str, **context: Any):
        self.t_start = time.time()
        self.session = str(session)
        self.command = command
        self.run_id = f"{datetime.fromtimestamp(self.t_start, tz=timezone.utc):%Y%m%dT%H%M%S}Z-{os.getpid()}"
        day = datetime.fromtimestamp(self.t_start, tz=timezone.utc).strftime("%Y-%m-%d")
        self.path = Path(root) / "logs" / day / f"{self.session}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.event("run", "start", **context)

    def event(self, stage: str, event: str, **fields: Any) -> None:
        """Append one line (written and flushed at once, so a crash loses at most the line being written)."""
        now = time.time()
        line = {"time_utc": _utc_iso(now), "run_id": self.run_id, "session": self.session, "command": self.command,
                "stage": stage, "event": event, **fields}
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(to_json(line, indent=None) + "\n")

    @contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator[dict]:
        """`with log.stage("bank") as counts: ... counts["frames"] = n`: start, then end (wall time and
        counts) or error (stack trace; the exception is re-raised)."""
        counts: dict = {}
        t0 = time.time()
        self.event(name, "start", **fields)
        try:
            yield counts
        except BaseException as e:
            self.error(name, e, elapsed_s=time.time() - t0, **counts)
            raise
        self.event(name, "end", elapsed_s=time.time() - t0, **counts)

    def error(self, stage: str, exc: BaseException, **fields: Any) -> None:
        self.event(stage, "error", error_type=type(exc).__name__, error=str(exc),
                   traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), **fields)

    def finish(self, ok: bool, **fields: Any) -> None:
        """The run's last line: `end` (ok) or `failed`, with the total wall time."""
        self.event("run", "end" if ok else "failed", elapsed_s=time.time() - self.t_start, **fields)
