#!/bin/bash
#SBATCH --job-name=V_Qwen_Mixed_1e4_4gpu
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=15:00:00
#SBATCH --mem=480G
#SBATCH --output=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/train/multiple_gpus/r16/lr_1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/train/multiple_gpus/r16/lr_1e4_%j.err

set -e

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_x86/bin/activate

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

export HF_HOME="/path/to/your/project_root/.hf_cache"

BASE_MODEL="./models/Qwen2.5-7B-Instruct"
TRAIN="./Data/Mixed_Dataset/mixed_TRAIN.jsonl"
EVAL="./Data/Mixed_Dataset/mixed_DEV.jsonl"
OUT_DIR="./fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/checkpoints/Mixed_Baseline/multiple_gpus/r16/lr_1e4"
RANK=16
EPOCHS=5
NUM_GPUS=4

BATCH_SIZE=4
GRAD_ACCUM=1

mkdir -p "$OUT_DIR"

TRAIN_SCRIPT="fine_tuning/fine_tuning_sequential/vanilla/code/train.py"

for LR in 1.1247e-4; do
    echo "=========================================================="
    echo " Mixed Baseline | LR=${LR} | Rank=${RANK} | Epochs=${EPOCHS} | GPUs=${NUM_GPUS}"
    echo "=========================================================="
    torchrun --standalone --nproc_per_node=$NUM_GPUS "$TRAIN_SCRIPT" \
        --phase              1 \
        --base_model_path    "$BASE_MODEL" \
        --train_dataset_path "$TRAIN" \
        --eval_dataset_path  "$EVAL" \
        --output_dir         "$OUT_DIR/lr${LR}_rank${RANK}" \
        --search_lr          "$LR" \
        --search_rank        $RANK \
        --batch_size         $BATCH_SIZE \
        --grad_accum         $GRAD_ACCUM \
        --epochs             $EPOCHS
    echo ">>> LR=${LR} done <<<"
done

echo "=================================================================="
echo " Mixed Baseline (10^-4) complete. Results in: $OUT_DIR"
echo "=================================================================="
