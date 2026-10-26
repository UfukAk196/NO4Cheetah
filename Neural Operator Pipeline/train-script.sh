#!/bin/bash -l
#SBATCH --job-name=fno_cheetah_train
#SBATCH --output=fno_%J.out
#SBATCH --error=fno_%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=hpcgwgpu
#SBATCH --gres=gpu:1

source /etc/profile.d/modules.sh
module purge
module load maxwell pytorch

spack env activate pytorch

cd $SLURM_SUBMIT_DIR

pip install -e .[logging]

python scripts/train_model.py