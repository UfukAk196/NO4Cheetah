"""
Distributed Optuna hyperparameter search for the FNO of ``train_model.py``
(settings in ``scno.config``).

Minimizes the Sobolev (H1) test loss at the training resolution, the metric the
neuralop ``Trainer`` reports as ``"<train_resolution>_h1"``. Every trial trains with
``scno.train.fit`` (same code path as a normal training run) and settings drawn by
``suggest``; unpromising trials are stopped early by the Hyperband pruner, which sees
the metric at every evaluation.

Any number of workers (processes, Slurm array tasks on several nodes) share one study
through a journal file on the shared filesystem; ``maxwell-script.sh`` submits them on
Maxwell. The study is a run ``<RUN>`` like a training run and writes only:
    tables/<RUN>/optuna_journal.log   the study (all trials), shared by the workers
    tables/<RUN>/trials.csv           one row per trial (``report``)
    figures/<RUN>/optuna_*.png        history, intermediate values, importances,
                                      parallel coordinates (``report``)
No model is saved per trial: copy the best parameters into ``scno.config`` and train.

Call:
    python scripts/tune_model.py create       new study, prints its run ID
    python scripts/tune_model.py generate     generate missing data (once, before workers)
    python scripts/tune_model.py work RUN     one worker, until N_TRIALS trials are done
    python scripts/tune_model.py report [RUN] table + plots (default: the latest study)
"""

import sys

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from scno.config import (
    DATA,
    DEVICE,
    FIGURES_DIR,
    LOSS,
    MODEL,
    N_TRIALS,
    RUN_DIRS,
    SEED,
    TABLES_DIR,
    TRAINING,
    TUNE_EPOCHS,
    TUNE_EVAL_INTERVAL,
    WANDB,
)
from scno.data.dataset import load_cheetah_sc
from scno.evaluate import _save_figures
from scno.train import HistoryTrainer, fit
from scno.utils.common import existing_runs, new_run_id, pick_device, save_csv, set_seed

METRIC = f"{DATA['train_resolution']}_h1"


def journal_path(run: str):
    return TABLES_DIR / run / "optuna_journal.log"


def load_study(run: str, create: bool = False) -> optuna.Study:
    """The study ``run`` in its journal file.

    The open-file lock works on network filesystems (Maxwell's GPFS/BeeGFS) and on
    Windows, unlike the default symlink lock. Sampler and pruner are not stored in
    the journal, so every worker builds them here. ``constant_liar`` keeps parallel
    TPE workers from sampling the same point while trials are still running.
    """
    path = journal_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path)))
    )
    kwargs = dict(
        study_name=run,
        storage=storage,
        sampler=optuna.samplers.TPESampler(constant_liar=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=TUNE_EVAL_INTERVAL, max_resource=TUNE_EPOCHS
        ),
    )
    if create:
        return optuna.create_study(direction="minimize", **kwargs)
    return optuna.load_study(**kwargs)


def suggest(trial: optuna.Trial) -> tuple[dict, dict, dict]:
    """Search space: ``MODEL``, ``LOSS`` and ``TRAINING`` with the tuned entries replaced.

    ``n_modes`` is capped at the Nyquist limit of the training grid; the gradient weight
    is tuned because the kick is the gradient of phi (``physics_weight`` stays 0).
    """
    n_modes = trial.suggest_int("n_modes", 4, DATA["train_resolution"] // 2)
    model_kwargs = {
        **MODEL,
        "n_modes": (n_modes,) * 3,
        "hidden_channels": trial.suggest_categorical("hidden_channels", [16, 32, 64]),
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "domain_padding": trial.suggest_categorical(
            "domain_padding", [0.0, 0.125, 0.25, 0.5]
        ),
    }
    loss_kwargs = {
        **LOSS,
        "gradient_weight": trial.suggest_float("gradient_weight", 0.1, 10.0, log=True),
    }
    training = {
        **TRAINING,
        # +1: the Trainer evaluates at epochs 0, k, 2k, ..., so the last epoch is one too.
        "epochs": trial.suggest_int("epochs", 1.0e3, 1.0e5, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1.0e-4, 1.0e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1.0e-6, 1.0e-3, log=True),
    }
    # other hyperparams to be tuned: regulization technique, loss function (Sobolev vs MAE vs MSE),
    # drop-out rate (Neurons/ hidden layers or data), data size, grid size, batch size, batch normalization parameters,
    # normalization/ scaling function, activation functions, optimization algorithm
    return model_kwargs, loss_kwargs, training


class PruningTrainer(HistoryTrainer):
    """``HistoryTrainer`` that reports ``METRIC`` to the trial at every evaluation and
    stops the training (``optuna.TrialPruned``) when the pruner says so."""

    def __init__(self, *args, trial: optuna.Trial, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial

    def log_eval(self, epoch, eval_metrics):
        super().log_eval(epoch=epoch, eval_metrics=eval_metrics)
        self.trial.report(float(eval_metrics[METRIC]), epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def work(run: str) -> None:
    device = pick_device(DEVICE)
    # Loaded once per worker, not per trial. generate=False: the data exists already
    # (`generate`), parallel workers must not write the same HDF5 files.
    train_loader, test_loaders, data_processor = load_cheetah_sc(
        **{**DATA, "generate": False}
    )
    # Tuning only needs the training resolution; the others would only cost time.
    res = DATA["train_resolution"]
    loaders = (train_loader, {res: test_loaders[res]}, data_processor)

    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED + trial.number)
        model_kwargs, loss_kwargs, training = suggest(trial)
        _, trainer = fit(
            loaders,
            device,
            model_kwargs,
            loss_kwargs,
            training,
            TUNE_EVAL_INTERVAL,
            nondim=DATA["nondim"],
            wandb_log=WANDB,
            trainer_cls=PruningTrainer,
            trial=trial,
        )
        return trainer.history["eval"][METRIC][-1]

    load_study(run).optimize(
        objective,
        callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED))],
    )


def report(run: str) -> None:
    study = load_study(run)
    trials = study.trials
    names = sorted({name for t in trials for name in t.params})
    rows = [
        {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "last_epoch": t.last_step,
            "duration_s": t.duration.total_seconds() if t.duration else None,
            **{name: t.params.get(name) for name in names},
        }
        for t in trials
    ]
    print(f"Trials saved: {save_csv(rows, TABLES_DIR / run / 'trials.csv')}")

    plots = optuna.visualization.matplotlib
    axes = {
        "optuna_history": plots.plot_optimization_history(study),
        "optuna_intermediate": plots.plot_intermediate_values(study),
        "optuna_parallel": plots.plot_parallel_coordinate(study),
    }
    if len(study.get_trials(states=(TrialState.COMPLETE,))) > 1:  # importances need >= 2
        axes["optuna_importances"] = plots.plot_param_importances(study)
        print(f"Best {METRIC} = {study.best_value:.4g} (trial {study.best_trial.number}):")
        print(f"  {study.best_params}")
    _save_figures({name: ax.figure for name, ax in axes.items()}, save_dir=FIGURES_DIR / run)


def main(command: str, run: str | None = None) -> int:
    if command == "create":
        run = new_run_id(*RUN_DIRS)
        load_study(run, create=True)
        print(run)  # last line of stdout: read by maxwell-script.sh
    elif command == "generate":
        load_cheetah_sc(**DATA)
    elif command == "work":
        work(run)
    elif command == "report":
        if run is None:
            runs = [r for r in existing_runs(*RUN_DIRS) if journal_path(r).exists()]
            if not runs:
                raise FileNotFoundError("No study found, run `tune_model.py create` first.")
            run = runs[-1]
        report(run)
    else:
        raise ValueError(f"Unknown command {command!r}, see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
