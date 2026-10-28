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
# Check the request without submitting:  sbatch --test-only tune-script.sh dummy
# Follow the jobs (sparingly, the info commands load the Slurm server):
#   squeue -u $USER          sacct -j <jobid>          scancel <jobid>
# Results when done:  python scripts/tune_model.py report <RUN>

set -euo pipefail                   # abort on error, unset variable, or any failing pipeline stage

source /etc/profile.d/modules.sh        # makes the `module` command available in a batch shell
module purge                            # drop inherited modules, so the run starts from a known state
module load maxwell pytorch conda/3.9   # site base paths + the cluster's CUDA PyTorch (pip must not replace it)

. conda-init
conda activate no4cheetah

export MPLBACKEND=Agg               # headless matplotlib: plt.show() must not block on a compute node

cd "$SLURM_SUBMIT_DIR"              # run from the submit dir, so the relative paths below resolve

if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${1:-}" ]; then # empty => login node; set => we are already a batch job
                                    # Install once, on the login node: N array tasks must not run pip into the same env at once.
    RUN=$(python scripts/tune_model.py create | tail -n 1)                      # new study, prints its run ID last
    GEN=$(sbatch --parsable --partition=maxgpu --constraint=A100 --nodes=1 \
        --time=12:00:00 --job-name=fno-gen --output=fno-gen-%j.out \
        --wrap "python scripts/tune_model.py generate")                         # generate the HDF5 data exactly once
    sbatch --dependency=afterok:"$GEN" --array=0-$(( ${1:-4} - 1 )) "$0" "$RUN" # N workers, held until data exists
    echo "Study $RUN: data job $GEN, workers wait for it."
    exit 0                          # stop here, or the login node would run `work` below
fi

echo "Worker started with SLURM_JOB_ID=$SLURM_JOB_ID, args=$*"
python scripts/tune_model.py work "${1:-}"   # batch job: one Optuna worker on study "$1"