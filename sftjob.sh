#!/bin/bash
#
#SBATCH --job-name=napierqwen_claric_sft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load apps/anaconda3/2023.03
conda init
source ~/.bashrc
conda activate sip_01
export MPLCONFIGDIR="/mnt/scratch/users/40645696/.cache_in_shared_scratch/matplotlib"
export TRANSFORMERS_CACHE="/mnt/scratch/users/40645696/.cache_in_shared_scratch/huggingface"
export HF_HOME="/mnt/scratch/users/40645696/.cache_in_shared_scratch/huggingface"

echo "Job started..."
python --version
cd /mnt/scratch/users/40645696/LLaMA-Factory
result=$(llamafactory-cli train examples/extras/nlg_eval/qwen25_qlora_predict.yaml)
echo "The python script output is: $result"
echo "Hello Error" 1>&2
