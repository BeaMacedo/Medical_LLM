#!/bin/bash
#SBATCH --job-name=V_Qwen_P2_Medicines_1e6
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=45:00:00
#SBATCH --mem=128G
#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/train/r64/lr_1e6_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/train/r64/lr_1e6_%j.err

set -e

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_x86/bin/activate

export HF_HOME="/path/to/your/project_root/.hf_cache"

BASE_MODEL="./models/Qwen2.5-7B-Instruct"
TRAIN="./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/clean/Finetuning_medicines_TRAIN.jsonl"
EVAL="./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/clean/Finetuning_medicines_DEV.jsonl"
PREV_ADAPTER="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase1_medqa/r8/lr_1e4/lr6.3246e-4_rank8/phase_1_final_adapter"
OUT_DIR="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase2_medicines/r64/lr_1e6"
RANK=64
EPOCHS=5

mkdir -p "$OUT_DIR"

TRAIN_SCRIPT="fine_tuning/fine_tuning_sequential/vanilla/code/train.py"

for LR in 3.5566e-6; do
    echo "=========================================================="
    echo " Phase 2 | LR=${LR} | Rank=${RANK} | Epochs=${EPOCHS}"
    echo "=========================================================="
    python "$TRAIN_SCRIPT" \
        --phase              2 \
        --base_model_path    "$BASE_MODEL" \
        --train_dataset_path "$TRAIN" \
        --eval_dataset_path  "$EVAL" \
        --prev_adapter_path  "$PREV_ADAPTER" \
        --output_dir         "$OUT_DIR/lr${LR}_rank${RANK}" \
        --search_lr          "$LR" \
        --search_rank        $RANK \
        --epochs             $EPOCHS
    echo ">>> LR=${LR} done <<<"
done

echo "=================================================================="
echo " Phase 2 (10^-6) complete. Results in: $OUT_DIR"
echo "=================================================================="
