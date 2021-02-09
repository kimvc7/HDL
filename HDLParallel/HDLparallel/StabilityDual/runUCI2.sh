#!/bin/bash

#SBATCH --job-name=HDL
#SBATCH --output=out_%a.txt
#SBATCH --error=err_%a.txt
#SBATCH -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=2G

#SBATCH --array=0-191

module load python/3.6.3
module load sloan/python/modules/3.6
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 17 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 18 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 19 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 20 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 21 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 22 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 23 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 24 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 25 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 26 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 27 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 28 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 29 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 30 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 31 --data_set uci8
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 32 --data_set uci8
