"""Small helpers: seeds, device, CSV tables, run IDs.

Parameter counts come from neuraloperator (``neuralop.utils.count_model_params``).
"""

from __future__ import annotations

import csv
import random
import re
from datetime import date
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_csv(rows: list[dict], path: str | Path) -> Path:
    """Writes ``rows`` (dicts with the same keys) as a CSV table with a header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return path


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
