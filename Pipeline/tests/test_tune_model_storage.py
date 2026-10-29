"""End-to-end check of the real ``tune_model`` storage functions.

Runs ``create`` -> trials -> ``report`` against a temporary run folder, so it exercises
``db_path``, ``storage_url``, ``_enable_wal`` and ``load_study`` as the script uses them.
"""

import tempfile
from pathlib import Path

import optuna

import scripts.tune_model as tm


def test_create_then_report_sees_trials(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(tm, "TABLES_DIR", tmp)

    study = tm.load_study("2026-10-29_01", create=True)
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=2)

    # `report` reopens the study from disk; it must see both trials.
    reopened = tm.load_study("2026-10-29_01")
    assert len(reopened.trials) == 2, len(reopened.trials)
    assert tm.db_path("2026-10-29_01").exists()
    assert tm.storage_url("2026-10-29_01").startswith("sqlite:///")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
