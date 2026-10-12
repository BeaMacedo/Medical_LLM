#!/bin/bash
#SBATCH --job-name=MedMCQA_zs
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00

#SBATCH --output=baselines_evaluation/logs/Qwen2.5-7B/zeroshot/%x_%j.out
#SBATCH --error=baselines_evaluation/logs/Qwen2.5-7B/zeroshot/%x_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

mkdir -p baselines_evaluation/results/Qwen2.5-7B/zeroshot
mkdir -p baselines_evaluation/logs/Qwen2.5-7B/zeroshot

python baselines_evaluation/eval_engine.py \
    --model_name "Qwen2.5-7B" \
    --dataset_name "MedMCQA" \
    --model_path "./models/Qwen2.5-7B" \
    --dataset_path "./Data/MedicalQA/Data_Generation/MedMCQA/Test/medmcqa_test.jsonl" \
    --output_dir "./baselines_evaluation/results/Qwen2.5-7B/zeroshot" \
    --answer_type "letters" \
    --max_length 8000 
