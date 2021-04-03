#!/bin/bash
  
#SBATCH --job-name=HDL
#SBATCH --output=out_%a.txt
#SBATCH --error=err_%a.txt
#SBATCH -p normal
#SBATCH --constraint=xeon-g6
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-20:00:00
#SBATCH --mem-per-cpu=4G

#SBATCH --array=0-71

source /etc/profile ; module load anaconda/2020a

python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 25 --data_set uci1
python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 26 --data_set uci1
python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 27 --data_set uci1
python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 28 --data_set uci1
python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 29 --data_set uci1
python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 30 --data_set uci1
python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 31 --data_set uci1
python -u trainUCI.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 32 --data_set uci1
