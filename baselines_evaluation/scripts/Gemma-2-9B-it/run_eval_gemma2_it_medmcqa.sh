#!/bin/bash
#SBATCH --job-name=Gemma2_it_MedMCQA_zs
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

#SBATCH --output=baselines_evaluation/logs/Gemma-2-9B-it/zeroshot/%x_%j.out
#SBATCH --error=baselines_evaluation/logs/Gemma-2-9B-it/zeroshot/%x_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

mkdir -p baselines_evaluation/results/Gemma-2-9B-it

python baselines_evaluation/eval_engine.py \
    --model_name "Gemma-2-9B-it" \
    --dataset_name "MedMCQA" \
    --model_path "./models/Gemma-2-9B-it" \
    --dataset_path "./Data/MedicalQA/Data_Generation/MedMCQA/Test/medmcqa_test.jsonl" \
    --output_dir "./baselines_evaluation/results/Gemma-2-9B-it/zeroshot" \
    --answer_type "letters" \
    --max_length 8000 \
    --is_instruct
