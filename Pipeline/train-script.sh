#!/bin/bash
#SBATCH --job-name=fno_cheetah_train
#SBATCH --output=fno_train_%J.out
#SBATCH --error=fno_train_%J.err
#SBATCH --partition=maxgpu
#SBATCH --constraint=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

# usage with
#    sbatch train-script.sh

source /etc/profile.d/modules.sh
module purge
module load maxwell conda/3.9          # pytorch-Modul raus
. conda-init
conda activate no4cheetah

PY="$CONDA_PREFIX/bin/python"          # Python fest verdrahten, egal was im PATH rumfliegt
echo "Python: $PY"
"$PY" -c "import optuna, torch; print(optuna.__version__, torch.__version__, torch.cuda.is_available())"

export MPLBACKEND=Agg               # headless matplotlib: plt.show() must not block on a compute node

cd "$SLURM_SUBMIT_DIR"             # run from the submit dir, so the relative paths below resolve

"$PY" scripts/train_model.py      # train, generating missing data first