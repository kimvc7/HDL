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

python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 16 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 17 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 18 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 19 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 20 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 21 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 22 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 23 --data_set uci1
python -u trainCIFAR.py -m ff --gnum ${SLURM_ARRAY_TASK_ID} --mnum 24 --data_set uci1
