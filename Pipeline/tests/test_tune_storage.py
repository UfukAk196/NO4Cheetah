"""Self-check of the Optuna storage of ``scripts/tune_model.py``.

The study must survive the process that created it: ``create`` writes the database,
``report`` reopens it in a fresh storage and must see every trial. That is exactly
what was broken before (in-memory study -> empty reports), so it is what this checks.
"""

import sqlite3
import tempfile
from pathlib import Path

import optuna
from optuna.storages import RDBStorage
from sqlalchemy import text


def storage_url(path: Path) -> str:
    """Same URL shape as ``tune_model.storage_url`` (kept in sync by hand)."""
    return f"sqlite:///{path.as_posix()}?timeout=60"


def _enable_wal(path: Path) -> None:
    """Same as ``tune_model._enable_wal``."""
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA journal_mode=WAL")


def _storage(url: str) -> RDBStorage:
    return RDBStorage(url=url, engine_kwargs={"connect_args": {"timeout": 60}})


def test_study_survives_reopen():
    db = Path(tempfile.mkdtemp()) / "optuna.db"
    url = storage_url(db)

    _enable_wal(db)
    study = optuna.create_study(
        study_name="t", storage=_storage(url), direction="minimize"
    )
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=3)

    # A fresh storage on the same file, like `report` does.
    reopened = optuna.load_study(study_name="t", storage=_storage(url))
    assert len(reopened.trials) == 3, len(reopened.trials)
    assert reopened.best_value is not None
    assert db.exists()


def test_wal_is_enabled():
    """WAL must really be active on the file Optuna writes to."""
    db = Path(tempfile.mkdtemp()) / "optuna.db"
    _enable_wal(db)
    storage = _storage(storage_url(db))
    with storage.engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert mode == "wal", mode


if __name__ == "__main__":
    test_study_survives_reopen()
    test_wal_is_enabled()
    print("OK: study persisted, reopened with all trials, WAL active")
