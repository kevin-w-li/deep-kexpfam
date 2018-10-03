#!/bin/bash

#SBATCH --cpus-per-task=6
#SBATCH --partition=compute
#SBATCH --array=0-74

source ~/anaconda/etc/profile.d/conda.sh
conda activate py2

set -eu

dsets=( redwine whitewine parkinsons hepmass miniboone )
SEED=$(( $SLURM_ARRAY_TASK_ID % 15 ))
DSET=${dsets[$(( $SLURM_ARRAY_TASK_ID / 15 ))]}

cd ~/deep-kexpfam
python bias_est.py --dset $DSET --seed $SEED --cpu-count 6
