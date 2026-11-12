#!/bin/bash
#SBATCH --job-name=fno_cheetah_train
#SBATCH --output=scripts/fno_train_%J.out
#SBATCH --error=scripts/fno_train_%J.err
#SBATCH --partition=maxgpu
#SBATCH --constraint=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

# usage, from Pipeline/ (logs go to scripts/, relative to the submit dir):
#    sbatch scripts/train-script.sh

source /etc/profile.d/modules.sh
module purge
module load maxwell conda/3.9          # pytorch-Modul raus
. conda-init
conda activate no4cheetah

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$CONDA_PREFIX/bin/python"          # Python fest verdrahten, egal was im PATH rumfliegt
echo "Python: $PY"
"$PY" -c "import optuna, torch; print(optuna.__version__, torch.__version__, torch.cuda.is_available())"

export MPLBACKEND=Agg               # headless matplotlib: plt.show() must not block on a compute node

cd "$SLURM_SUBMIT_DIR"             # run from the submit dir, so the relative paths below resolve
[ -f exec/train_model.py ] || { echo "Submit from Pipeline/: sbatch scripts/train-script.sh" >&2; exit 1; }

"$PY" exec/train_model.py      # train, generating missing data first