"""
Distributed Optuna hyperparameter search for the TFNO of ``train_model.py``.

Minimizes the H1 test loss at the training resolution, the metric the neuralop
``Trainer`` reports as ``"<train_resolution>_h1"``. Every trial builds TFNO, AdamW and
the ``Trainer`` like ``train_model.py``, with the settings drawn by ``suggest``;
unpromising trials are stopped early by the Hyperband pruner, which sees the metric at
every evaluation. All settings are at the top of this file.

Any number of workers (processes, Slurm array tasks on several nodes) share one study
through a journal file on the shared filesystem; ``scripts/tune-script.sh`` submits them
on Maxwell. The study is a run ``<RUN>`` = ``<DATE>_<NN>`` like a training run and writes
only to ``outputs/studies/optuna/<RUN>/``:
    <DATE>_optuna_journal_<NN>.log   the study (all trials), shared by the workers
    <DATE>_trials_<NN>.csv           one row per trial (``report``)
    optuna_*.png                     history, intermediate values, importances,
                                     parallel coordinates (``report``)
No model is saved per trial: copy the best parameters into ``train_model.py`` and train.

Call:
    python exec/tune_model.py create       new study, prints its run ID
    python exec/tune_model.py generate     generate missing data (once, before workers)
    python exec/tune_model.py work RUN     one worker, until N_TRIALS trials are done
    python exec/tune_model.py report [RUN] table + plots (default: the latest study)
"""

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from neuralop import H1Loss, LpLoss, Trainer
from neuralop.models import TFNO
from neuralop.training import AdamW
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from scno.config import OPTUNA_DIR, RUN_DIRS, DATA, SEED
from scno.data.dataset import load_cheetah_sc
from scno.evaluate import _save_figures
from scno.utils.common import existing_runs, new_run_id, print_data_settings

N_TRIALS = 100  # finished (complete + pruned) trials over all workers
TUNE_EPOCHS = 200  # max_resource of the Hyperband pruner
TUNE_EVAL_INTERVAL = 10  # epochs between evaluations = pruning checks

# Data as in train_model.py

METRIC = f"{DATA['train_resolution']}_h1"


def study_file(run: str, name: str) -> Path:
    """``studies/optuna/<RUN>/<DATE>_<stem>_<NN><suffix>`` for ``name`` = ``<stem><suffix>``."""
    name = Path(name)
    return OPTUNA_DIR / run / f"{run}_{name.name}"


def journal_path(run: str) -> Path:
    return study_file(run, "optuna_journal.log")


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


def suggest(trial: optuna.Trial) -> tuple[dict, dict]:
    """Search space: TFNO kwargs and training settings.

    ``n_modes`` is capped at the Nyquist limit of the training grid.
    """
    n_modes = trial.suggest_int("n_modes", 3, DATA["train_resolution"] // 2)
    model_kwargs = {
        "in_channels": 1,
        "out_channels": 1,
        "n_modes": (n_modes,) * 3,
        "hidden_channels": trial.suggest_categorical("hidden_channels", [2, 3, 5, 8, 10, 12, 16, 32, 64]),
        "n_layers": trial.suggest_int("n_layers", 1, 50),
        "domain_padding": trial.suggest_categorical(
            "domain_padding", [0.0, 0.01, 0.125, 0.25, 0.5, 1.0]
        ),
    }
    training = {
        "epochs": trial.suggest_int("epochs", 1.0e3, 1.0e5, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1.0e-4, 1.0e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1.0e-6, 1.0e-2, log=True),
    }
    # other hyperparams to be tuned: regulization technique, loss function (Sobolev vs MAE vs MSE),
    # drop-out rate (Neurons/ hidden layers or data), data size, grid size, batch size, batch normalization parameters,
    # normalization/ scaling function, activation functions, optimization algorithm
    return model_kwargs, training


def work(run: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Loaded once per worker, not per trial. The data exists already (`generate`), so
    # load_cheetah_sc does not write anything here.
    print_data_settings(DATA)
    train_loader, test_loaders, data_processor = load_cheetah_sc(**DATA)
    data_processor = data_processor.to(device)
    h1loss = H1Loss(d=3, periodic_in_x=False, periodic_in_y=False, periodic_in_z=False)
    l2loss = LpLoss(d=3, p=2)

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(SEED + trial.number)
        model_kwargs, training = suggest(trial)
    
        model = TFNO(**model_kwargs).to(device)
        optimizer = AdamW(
            model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        trainer = Trainer(
            model=model,
            n_epochs=training["epochs"],
            device=device,
            data_processor=data_processor,
            wandb_log=False,
            eval_interval=TUNE_EVAL_INTERVAL,
            use_distributed=False,
            verbose=True,  # log_eval is only called with verbose=True
        )

        # Pruning: the Trainer's own log_eval is called as usual, then METRIC is
        # reported to the trial, which may stop the training.
        log_eval, last = trainer.log_eval, {}

        def report_eval(epoch, eval_metrics):
            log_eval(epoch=epoch, eval_metrics=eval_metrics)
            last[METRIC] = float(eval_metrics[METRIC])
            trial.report(last[METRIC], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trainer.log_eval = report_eval
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            training_loss=h1loss,
            eval_losses={"h1": h1loss, "l2": l2loss},
        )
            
        return last[METRIC]

    load_study(run).optimize(
        objective,
        callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED))],
    )

    report(run)


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
    path = study_file(run, "trials.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Trials saved: {path}")

    plots = optuna.visualization.matplotlib
    axes = {
        "optuna_rank": plots.plot_rank(study),
        "optuna_history": plots.plot_optimization_history(study),
        "optuna_intermediate": plots.plot_intermediate_values(study),
        "optuna_parallel": plots.plot_parallel_coordinate(study),
    }
    if len(study.get_trials(states=(TrialState.COMPLETE,))) > 1:  # importances need >= 2
        axes["optuna_importances"] = plots.plot_param_importances(study)
        print(f"Best {METRIC} = {study.best_value:.4g} (trial {study.best_trial.number}):")
        print(f"  {study.best_params}")

    def _figure_of(ax):
        # plot_rank/plot_slice/plot_contour return an ndarray of Axes for >1 param
        return np.ravel(ax)[0].figure

    _save_figures({name: _figure_of(ax) for name, ax in axes.items()}, save_dir=OPTUNA_DIR / run)


def main(command: str, run: str | None = None) -> int:
    if command == "create":
        run = new_run_id(*RUN_DIRS)
        load_study(run, create=True)
        print(run)  # last line of stdout: read by scripts/tune-script.sh
    elif command == "generate":
        load_cheetah_sc(**DATA)
    elif command == "work":
        work(run)
    elif command == "report":
        if run is None:
            runs = [r for r in existing_runs(*RUN_DIRS) if journal_path(r).exists()]
            if not runs:
                raise FileNotFoundError("No study found, run `tune_model.py create` first.")
            for r in runs:
                print(f"Study {r}:")
                report(r)
                print()
        else:
            report(run)
    else:
        raise ValueError(f"Unknown command {command!r}, see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
