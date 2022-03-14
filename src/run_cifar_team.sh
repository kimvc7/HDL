#!/bin/bash
  
#SBATCH --job-name=MNIST
#SBATCH --output=outputs/MNIST_outputs/out_%a.txt
#SBATCH --error=outputs/MNIST_errors/err_%a.txt
#SBATCH -p normal
#SBATCH --constraint=xeon-g6
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-8:30:00
#SBATCH --mem-per-cpu=4G

#SBATCH --array=0-95

source /etc/profile ; 
module load anaconda/2021a

python train.py --data_set mnist --network_type MLP --exp_id ${SLURM_ARRAY_TASK_ID}
