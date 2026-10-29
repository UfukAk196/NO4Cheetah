#!/bin/bash -l
#SBATCH --job-name=fno-tune
#SBATCH --output=fno-tune-%A_%a.out
#SBATCH --error=fno-tune-%A_%a.err
#SBATCH --partition=maxgpu
#SBATCH --constraint=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00

# Distributed Optuna search for the FNO (scripts/tune_model.py) on Maxwell.
# All #SBATCH lines must stay above the first command, otherwise Slurm ignores them.
# Narrow the GPU type with e.g. --constraint="A100" (see `savail` / `sinfo`).
#
# Usage, on a Maxwell login node from "Neural Operator Pipeline/":
#   bash tune-script.sh [N_WORKERS]      (default 4)
# This creates the study on the login node, submits one data-generation job and then an
# array of N_WORKERS Optuna workers that start only after the data exists (afterok), so
# the workers never write the same HDF5 files. Every array task re-runs this script
# under sbatch and becomes one worker sharing tables/<RUN>/optuna_journal.log.
#
# Check the request without submitting:  sbatch --test-only --export=ALL,TUNE_RUN=dummy tune-script.sh
# Follow the jobs (sparingly, the info commands load the Slurm server):
#   squeue -u $USER          sacct -j <jobid>          scancel <jobid>
# Results when done:  python scripts/tune_model.py report <RUN>

set -euo pipefail                   # abort on error, unset variable, or any failing pipeline stage

# Not SLURM_JOB_ID / SLURM_SUBMIT_DIR: a JupyterHub terminal on Maxwell is itself a Slurm job
# (submit dir /etc/jupyterhub). Workers start in --chdir, so $PWD is right in both cases.

PIPELINE_DIR="$PWD"                 # saved before module/conda setup, which may change the cwd

source /etc/profile.d/modules.sh
module purge
module load maxwell conda/3.9          # pytorch-Modul raus
. conda-init
conda activate no4cheetah

PY="$CONDA_PREFIX/bin/python"          # Python fest verdrahten, egal was im PATH rumfliegt
echo "Python: $PY"
"$PY" -c "import optuna, torch; print(optuna.__version__, torch.__version__, torch.cuda.is_available())"

export MPLBACKEND=Agg               # headless matplotlib: plt.show() must not block on a compute node

cd "$PIPELINE_DIR"                  # run from "Neural Operator Pipeline/", so the relative paths below resolve

if [ -z "${TUNE_RUN:-}" ]; then    # unset => started by you; set => one of the worker jobs submitted below
    RUN=$(python scripts/tune_model.py create | tail -n 1)                      # new study, prints its run ID last
    GEN=$(sbatch --parsable --chdir="$PIPELINE_DIR" --partition=maxgpu --constraint=A100 --nodes=1 \
        --time=12:00:00 --job-name=fno-gen --output=fno-gen-%j.out \
        --wrap "$PY scripts/tune_model.py generate")                         # generate the HDF5 data exactly once
    sbatch --chdir="$PIPELINE_DIR" --export=ALL,TUNE_RUN="$RUN" \
        --dependency=afterok:"$GEN" --array=0-$(( ${1:-4} - 1 )) \
        "$PIPELINE_DIR/tune-script.sh"                                          # N workers, held until data exists
    echo "Study $RUN: data job $GEN, workers wait for it."
    exit 0                          # stop here, or this shell would run `work` below
fi

echo "Worker started with SLURM_JOB_ID=${SLURM_JOB_ID:-}, TUNE_RUN=$TUNE_RUN"
"$PY" scripts/tune_model.py work "$TUNE_RUN"   # batch job: one Optuna worker on study $TUNE_RUN