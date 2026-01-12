#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


module load apps/anaconda3/2023.03
conda init
source ~/.bashrc
module load libs/nvidia-cuda/12.2.2/bin
conda activate sip_01

echo "Hello World"
python --version
nvidia-smi
