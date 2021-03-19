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

python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 0 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 1 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 2 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 3 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 4 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 5 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 6 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 7 --data_set uci1
