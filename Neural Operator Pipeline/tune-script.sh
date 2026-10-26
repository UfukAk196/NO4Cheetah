#!/bin/bash
#SBATCH --partition=maxgpu
#SBATCH --constraint=A100
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fno-tune
#SBATCH --output=fno-tune-%A_%a.out
# Distributed Optuna search for the FNO (scripts/tune_model.py) on Maxwell.
# All #SBATCH lines must stay above the first command, otherwise Slurm ignores them.
# Narrow the GPU type with e.g. --constraint="A100" (see `savail` / `sinfo`).
#
# Usage, on a Maxwell login node from "Neural Operator Pipeline/":
#   bash maxwell-script.sh [N_WORKERS]      (default 4)
# This creates the study on the login node, submits one data-generation job and then an
# array of N_WORKERS Optuna workers that start only after the data exists (afterok), so
# the workers never write the same HDF5 files. Every array task re-runs this script
# under sbatch and becomes one worker sharing tables/<RUN>/optuna_journal.log.
#
# Check the request without submitting:  sbatch --test-only maxwell-script.sh dummy
# Follow the jobs (sparingly, the info commands load the Slurm server):
#   squeue -u $USER          sacct -j <jobid>          scancel <jobid>
# Results when done:  python scripts/tune_model.py report <RUN>
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate no4cheetah-gpu
export MPLBACKEND=Agg

if [ -z "${SLURM_JOB_ID:-}" ]; then
    RUN=$(python scripts/tune_model.py create | tail -n 1)
    GEN=$(sbatch --parsable --partition=maxgpu --constraint=GPU --nodes=1 \
        --time=12:00:00 --job-name=fno-gen --output=fno-gen-%j.out \
        --wrap "python scripts/tune_model.py generate")
    sbatch --dependency=afterok:"$GEN" --array=0-$(( ${1:-4} - 1 )) "$0" "$RUN"
    echo "Study $RUN: data job $GEN, workers wait for it."
    exit 0
fi

python scripts/tune_model.py work "$1"
