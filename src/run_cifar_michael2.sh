#!/bin/bash

#SBATCH -o output.sh.log-%j
#SBATCH --error=error.sh.log-%j
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-23:30:00
#SBATCH --mem-per-cpu=4G

#SBATCH --array=200-399

source /etc/profile ;
module load anaconda/2021a

python train.py --train_size 0.1 --data_set cifar10 --network_type ALEX --exp_id ${SLURM_ARRAY_TASK_ID}
