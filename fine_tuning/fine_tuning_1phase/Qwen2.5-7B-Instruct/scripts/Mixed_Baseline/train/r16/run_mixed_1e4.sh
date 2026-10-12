#!/bin/bash
#SBATCH --job-name=V_Qwen_Mixed_1e4
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=45:00:00
#SBATCH --mem=128G
#SBATCH --output=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/train/r16/lr_1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/train/r16/lr_1e4_%j.err

set -e

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_x86/bin/activate

export HF_HOME="/path/to/your/project_root/.hf_cache"

BASE_MODEL="./models/Qwen2.5-7B-Instruct"
TRAIN="./Data/Mixed_Dataset/mixed_TRAIN.jsonl"
EVAL="./Data/Mixed_Dataset/mixed_DEV.jsonl"
OUT_DIR="./fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/checkpoints/Mixed_Baseline/r16/lr_1e4"
RANK=16
EPOCHS=5

mkdir -p "$OUT_DIR"

TRAIN_SCRIPT="fine_tuning/fine_tuning_sequential/vanilla/code/train.py"

for LR in 1.1247e-4; do
    echo "=========================================================="
    echo " Mixed Baseline | LR=${LR} | Rank=${RANK} | Epochs=${EPOCHS}"
    echo "=========================================================="
    python "$TRAIN_SCRIPT" \
        --phase              1 \
        --base_model_path    "$BASE_MODEL" \
        --train_dataset_path "$TRAIN" \
        --eval_dataset_path  "$EVAL" \
        --output_dir         "$OUT_DIR/lr${LR}_rank${RANK}" \
        --search_lr          "$LR" \
        --search_rank        $RANK \
        --epochs             $EPOCHS
    echo ">>> LR=${LR} done <<<"
done

echo "=================================================================="
echo " Mixed Baseline (10^-4) complete. Results in: $OUT_DIR"
echo "=================================================================="
