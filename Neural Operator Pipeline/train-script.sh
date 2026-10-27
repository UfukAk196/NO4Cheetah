#!/bin/bash -l
#SBATCH --job-name=fno_cheetah_train
#SBATCH --output=fno_%J.out
#SBATCH --error=fno_%J.err
#SBATCH --partition=maxgpu
#SBATCH --constraint=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00


source /etc/profile.d/modules.sh   # makes the `module` command available in a batch shell
module purge                       # drop inherited modules, so the run starts from a known state
module load maxwell pytorch        # site base paths + the cluster's CUDA PyTorch (pip must not replace it)

spack env activate pytorch         # wire up the Spack install tree (PATH, LD_LIBRARY_PATH, PYTHONPATH)

cd "$SLURM_SUBMIT_DIR"             # run from the submit dir, so the relative paths below resolve

pip install --user -e ".[logging]" # editable install of scno + wandb; torch is left untouched
python scripts/train_model.py      # train, generating missing data first