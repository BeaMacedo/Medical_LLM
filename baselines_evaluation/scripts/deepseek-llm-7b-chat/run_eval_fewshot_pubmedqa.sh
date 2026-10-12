#!/bin/bash
#SBATCH --job-name=Deepseek_PubMedQA_fs
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

#SBATCH --output=baselines_evaluation/logs/deepseek-llm-7b-chat/fewshot/%x_%j.out
#SBATCH --error=baselines_evaluation/logs/deepseek-llm-7b-chat/fewshot/%x_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

mkdir -p baselines_evaluation/results/deepseek-llm-7b-chat/fewshot
mkdir -p baselines_evaluation/logs/deepseek-llm-7b-chat/fewshot

python baselines_evaluation/eval_engine_fewshot.py \
    --model_name "deepseek-llm-7b-chat" \
    --dataset_name "PubMedQA" \
    --model_path "./models/deepseek-llm-7b-chat" \
    --dataset_path "./Data/MedicalQA/Data_Generation/PubMedQA/Test_Dev/pubmedqa_test.jsonl" \
    --few_shot_path "./Data/MedicalQA/Data_Generation/PubMedQA/Test_Dev/few_shot_examples_pubmedqa.jsonl" \
    --output_dir "./baselines_evaluation/results/deepseek-llm-7b-chat/fewshot" \
    --answer_type "yes_no_maybe" \
    --max_length 8000 \
    --is_instruct
