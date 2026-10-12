#!/bin/bash
#SBATCH --job-name=Gemma2_it_MedQA_fs
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

#SBATCH --output=baselines_evaluation/logs/Gemma-2-9B-it/fewshot/%x_%j.out
#SBATCH --error=baselines_evaluation/logs/Gemma-2-9B-it/fewshot/%x_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

mkdir -p baselines_evaluation/results/Gemma-2-9B-it/fewshot

python baselines_evaluation/eval_engine_fewshot.py \
    --model_name "Gemma-2-9B-it" \
    --dataset_name "MedQA" \
    --model_path "./models/Gemma-2-9B-it" \
    --dataset_path "./Data/MedicalQA/Data_Generation/MedQA/Test/medqa_test.jsonl" \
    --few_shot_path "./Data/MedicalQA/Data_Generation/MedQA/Test/few_shot_examples_medqa.jsonl" \
    --output_dir "./baselines_evaluation/results/Gemma-2-9B-it/fewshot" \
    --answer_type "letters" \
    --max_length 8000 \
    --is_instruct
