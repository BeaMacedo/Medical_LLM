#!/bin/bash
#SBATCH --job-name=V_P1_MedQA_1e4
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Gemma_2_9b_it/phase1_medqa/train/r8/lr_1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Gemma_2_9b_it/phase1_medqa/train/r8/lr_1e4_%j.err

set -e  

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_x86/bin/activate

export HF_HOME="/path/to/your/project_root/.hf_cache"

BASE_MODEL="./models/Gemma-2-9B-it"
TRAIN="./Data/MedicalQA/Data_Generation/MedQA/Train/MedQA_train_final_UPDATED.jsonl"
EVAL="./Data/MedicalQA/Data_Generation/MedQA/Train/MedQA_dev_UPDATED.jsonl"
OUT_DIR="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Gemma_2_9b_it/phase1_medqa/r8/lr_1e4"
RANK=8
EPOCHS=5

mkdir -p "$OUT_DIR"

TRAIN_SCRIPT="fine_tuning/fine_tuning_sequential/vanilla/code/train.py"

for LR in 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4; do
    echo "=========================================================="
    echo " Phase 1 | LR=${LR} | Rank=${RANK} | Epochs=${EPOCHS}"
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
echo " Phase 1 (10^-4) complete. Results in: $OUT_DIR"
echo "=================================================================="
