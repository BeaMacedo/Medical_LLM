#!/bin/bash
#SBATCH --job-name=AdapR32_V_Qwen_P3_DDIs_1e4
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --mem=128G
#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase3_ddis/train/adapt_r32_35566e-5/r4/lr_1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase3_ddis/train/adapt_r32_35566e-5/r4/lr_1e4_%j.err

set -e

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_x86/bin/activate

BASE_MODEL="./models/Qwen2.5-7B-Instruct"
TRAIN="./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/final_TRAIN.jsonl"
EVAL="./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/final_DEV.jsonl"
PREV_ADAPTER="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase2_medicines/r32/lr_1e5/lr3.5566e-5_rank32/phase_2_final_adapter"
PREV_ADAPTER_PHASE1="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase1_medqa/r8/lr_1e4/lr6.3246e-4_rank8/phase_1_final_adapter"
OUT_DIR="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase3_ddis/adapt_r32_35566e-5/r4/lr_1e4"
RANK=4
EPOCHS=5

mkdir -p "$OUT_DIR"

TRAIN_SCRIPT="fine_tuning/fine_tuning_sequential/vanilla/code/train.py"

for LR in 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4; do
    echo "=========================================================="
    echo " Phase 3 | LR=${LR} | Rank=${RANK} | Epochs=${EPOCHS}"
    echo "=========================================================="
    python "$TRAIN_SCRIPT" \
        --phase                3 \
        --base_model_path      "$BASE_MODEL" \
        --train_dataset_path   "$TRAIN" \
        --eval_dataset_path    "$EVAL" \
        --prev_adapter_path    "$PREV_ADAPTER" \
        --prev_adapter_path_phase1 "$PREV_ADAPTER_PHASE1" \
        --output_dir           "$OUT_DIR/lr${LR}_rank${RANK}" \
        --search_lr            "$LR" \
        --search_rank          $RANK \
        --epochs               $EPOCHS \
        --grad_accum            1
    echo ">>> LR=${LR} done <<<"
done

echo "=================================================================="
echo " Phase 3 (10^-4) complete. Results in: $OUT_DIR"
echo "=================================================================="
