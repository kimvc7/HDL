#!/bin/bash

#SBATCH --job-name=HDL
#SBATCH --output=out_%a.txt
#SBATCH --error=err_%a.txt
#SBATCH -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=2G

#SBATCH --array=0-71

module load python/3.6.3
module load sloan/python/modules/3.6
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 0 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 1 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 2 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 3 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 4 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 5 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 6 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 7 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 8 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 9 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 10 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 11 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 12 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 13 --data_set uci58
xvfb-run -d python3.6 trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 14 --data_set uci58
