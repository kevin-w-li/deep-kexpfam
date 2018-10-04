#!/bin/bash

# don't do:  --cpus-per-task=20
#SBATCH --exclusive
#SBATCH --exclude=u436a
#SBATCH --partition=compute,wrkstn
#SBATCH --array=0-74

source ~/anaconda/etc/profile.d/conda.sh
conda activate py2

set -eu

dsets=( redwine whitewine parkinsons hepmass miniboone )
SEED=$(( $SLURM_ARRAY_TASK_ID % 15 ))
DSET=${dsets[$(( $SLURM_ARRAY_TASK_ID / 15 ))]}

cd ~/deep-kexpfam
nice python bias_est.py --dset $DSET --seed $SEED # --cpu-count 20
