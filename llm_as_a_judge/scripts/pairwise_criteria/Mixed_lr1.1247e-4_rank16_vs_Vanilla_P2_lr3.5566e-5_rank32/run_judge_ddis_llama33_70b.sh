#!/bin/bash
#SBATCH --job-name=Judge_DDIs_pairwise_Mixed_vs_SequentialP2_Llama33-70B
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

#SBATCH --output=llm_as_a_judge/logs/Llama-3.3-70B-Instruct/pairwise/%x_%j.out
#SBATCH --error=llm_as_a_judge/logs/Llama-3.3-70B-Instruct/pairwise/%x_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_judge/bin/activate

JSONL="./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/final_TEST_open.jsonl"

RESPONSE_A_CSV="fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/results/Mixed_Baseline/long_form/r16/lr_1e4/lr1.1247e-4_rank16/Mixed_lr1.1247e-4_rank16_DDIs_GoldenSet_OpenQA.csv"
RESPONSE_B_CSV="fine_tuning/fine_tuning_sequential/vanilla/results/Qwen2.5-7B-Instruct/phase2_medicines/long_form/r32/lr_1e5/lr3.5566e-5_rank32/Vanilla_P2_lr3.5566e-5_rank32_DDIs_GoldenSet_OpenQA.csv"

python llm_as_a_judge/judge_pairwise_generate.py \
    --csv_path_a "$RESPONSE_A_CSV" \
    --csv_path_b "$RESPONSE_B_CSV" \
    --jsonl_path "$JSONL" \
    --criteria_module "criteria_pairwise" \
    --run_type "pairwise_criteria" \
    --judge_model_name "Llama-3.3-70B-Instruct" \
    --judge_model_path "./models/Llama-3.3-70B-Instruct" \
    --is_instruct \
    --alternate_order \
    --max_new_tokens 2500 \
    --output_dir "llm_as_a_judge/results"
