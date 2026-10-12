#!/bin/bash
#SBATCH --job-name=Judge_Medicines_pairwise_Mixed_vs_Baseline_Gemma3-27B
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00

#SBATCH --output=llm_as_a_judge/logs/Gemma-3-27B-PT/pairwise/%x_%j.out
#SBATCH --error=llm_as_a_judge/logs/Gemma-3-27B-PT/pairwise/%x_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_judge/bin/activate

JSONL="./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/clean/all_openqa.jsonl"

RESPONSE_A_CSV="fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/results/Mixed_Baseline/long_form/r16/lr_1e4/lr1.1247e-4_rank16/Mixed_lr1.1247e-4_rank16_Medicines_GoldenSet_OpenQA.csv"
RESPONSE_B_CSV="baselines_evaluation/results/Qwen2.5-7B-Instruct/long_form_evaluation/Qwen2.5-7B-Instruct_Medicines_GoldenSet_OpenQA.csv"

python llm_as_a_judge/judge_pairwise_generate.py \
    --csv_path_a "$RESPONSE_A_CSV" \
    --csv_path_b "$RESPONSE_B_CSV" \
    --jsonl_path "$JSONL" \
    --dataset_name "Medicines" \
    --model_b_name "Qwen2.5-7B-Instruct" \
    --criteria_module "criteria_pairwise" \
    --run_type "pairwise_criteria" \
    --judge_model_name "Gemma-3-27B-PT" \
    --judge_model_path "./models/Gemma-3-27B-PT" \
    --no-is_instruct \
    --prefix_force_json \
    --alternate_order \
    --max_new_tokens 2500 \
    --output_dir "llm_as_a_judge/results"