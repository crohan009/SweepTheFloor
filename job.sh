#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=SegNet_Training
#SBATCH --mail-type=END
##SBATCH --mail-user=rc3232@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:2

module purge
module load python/intel/2.7.12
module load pytorch/0.2.0_1
module load scikit-image/intel/0.12.3
module load torchvision/0.1.8

python sample_seg.py
