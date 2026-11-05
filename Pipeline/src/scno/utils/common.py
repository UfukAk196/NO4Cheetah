"""Run IDs ``<YYYY-MM-DD>_<NN>`` for the output folders."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

_RUN_ID = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2,})$")


def existing_runs(*dirs: str | Path) -> list[str]:
    """All run IDs ``<YYYY-MM-DD>_<NN>`` that end an entry name (without suffix) in
    ``dirs``, sorted chronologically."""
    names = [p.stem for d in map(Path, dirs) if d.is_dir() for p in d.iterdir()]
    runs = {m.group(1) for name in names if (m := _RUN_ID.search(name))}
    return sorted(runs, key=lambda run: (run[:10], int(run[11:])))


def new_run_id(*dirs: str | Path) -> str:
    """Next free run ID of today in ``dirs``: ``<YYYY-MM-DD>_<NN>``, ``NN`` = 01, 02, ..."""
    today = date.today().isoformat()
    numbers = [int(run[11:]) for run in existing_runs(*dirs) if run.startswith(today)]
    return f"{today}_{max(numbers, default=0) + 1:02d}"
