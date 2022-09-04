#!/bin/bash

#SBATCH -o output.text
#SBATCH --error=error.txt
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-23:30:00
#SBATCH --mem-per-cpu=4G

#SBATCH --array=0-199

source /etc/profile ;
module load anaconda/2021a

python train.py --train_size 1 --data_set 31 --network_type MLP --exp_id ${SLURM_ARRAY_TASK_ID}
