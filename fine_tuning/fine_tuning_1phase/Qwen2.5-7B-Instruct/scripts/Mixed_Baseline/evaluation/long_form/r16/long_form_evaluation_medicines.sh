#!/bin/bash
#SBATCH --job-name=Mixed_OpenQA_1e4
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

#SBATCH --output=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/evaluation/long_form/r16/lr_1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/evaluation/long_form/r16/lr_1e4_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

BASE_MODEL="./models/Qwen2.5-7B-Instruct"
ADAPTER_PATH="./fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/checkpoints/Mixed_Baseline/r16/lr_1e4/lr1.1247e-4_rank16/phase_1_final_adapter"
MODEL_NAME="Mixed_lr1.1247e-4_rank16"

OUTPUT_DIR="./fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/results/Mixed_Baseline/long_form/r16/lr_1e4/lr1.1247e-4_rank16"
mkdir -p "$OUTPUT_DIR"

TEST_MEDICINES="./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/clean/all_openqa.jsonl"

python fine_tuning/eval_openqa_phase.py \
    --model_name "$MODEL_NAME" \
    --model_path "$BASE_MODEL" \
    --peft_model_path "$ADAPTER_PATH" \
    --dataset_name "Medicines" \
    --dataset_path $TEST_MEDICINES \
    --output_dir $OUTPUT_DIR \
    --is_instruct
