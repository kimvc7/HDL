#!/bin/bash

#SBATCH --job-name=HDL
#SBATCH --output=out_%a.txt
#SBATCH --error=err_%a.txt
#SBATCH -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=4G

#SBATCH --array=0-39

module load python/3.6.3
module load sloan/python/modules/3.6
xvfb-run -d python3.6 trainCIFAR.py -m ff --runnum ${SLURM_ARRAY_TASK_ID} --data_set uci18
