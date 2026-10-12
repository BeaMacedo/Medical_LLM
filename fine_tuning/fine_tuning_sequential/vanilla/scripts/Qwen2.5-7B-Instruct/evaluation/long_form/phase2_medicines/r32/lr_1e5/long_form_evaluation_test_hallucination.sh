#!/bin/bash
#SBATCH --job-name=V_P2_OpenQA_lr3.5566e-5_rank32
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/evaluation/long_form/r32/TESTE_lr_1e5_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/evaluation/long_form/r32/TESTE_lr_1e5_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

BASE_MODEL="./models/Qwen2.5-7B-Instruct"
PREV_ADAPTER="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase1_medqa/r8/lr_1e4/lr6.3246e-4_rank8/phase_1_final_adapter"
ADAPTER_PATH="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase2_medicines/r32/lr_1e5/lr3.5566e-5_rank32/phase_2_final_adapter"
MODEL_NAME="Vanilla_P2_lr3.5566e-5_rank32"

OUTPUT_DIR="./fine_tuning/fine_tuning_sequential/vanilla/results/Qwen2.5-7B-Instruct/phase2_medicines/long_form/r32/lr_1e5/lr3.5566e-5_rank32/medicines_TESTE_alucinacoes"
mkdir -p "$OUTPUT_DIR"

TEST_MEDICINES="./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/teste.jsonl"

python fine_tuning/eval_openqa_phase.py \
    --model_name "$MODEL_NAME" \
    --model_path "$BASE_MODEL" \
    --prev_adapters_paths "$PREV_ADAPTER" \
    --peft_model_path "$ADAPTER_PATH" \
    --dataset_name "Medicines" \
    --dataset_path $TEST_MEDICINES \
    --output_dir $OUTPUT_DIR \
    --is_instruct
