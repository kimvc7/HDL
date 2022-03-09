#!/bin/bash
  
#SBATCH --job-name=CIFAR
#SBATCH --output=outputs/CIFAR_outputs/out_%a.txt
#SBATCH --error=outputs/CIFAR_errors/err_%a.txt
#SBATCH -p normal
#SBATCH --constraint=xeon-g6
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-18:30:00
#SBATCH --mem-per-cpu=4G

#SBATCH --array=1-10

source /etc/profile ; 
module load anaconda/2021a

python train.py --data_set cifar10 --network_type ALEX --exp_id ${SLURM_ARRAY_TASK_ID}
