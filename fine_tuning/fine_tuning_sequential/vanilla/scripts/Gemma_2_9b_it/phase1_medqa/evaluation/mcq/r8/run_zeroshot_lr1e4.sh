#!/bin/bash
#SBATCH --job-name=V_P1_ZS_1e4
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=45:00:00
#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Gemma_2_9b_it/phase1_medqa/evaluation/mcq/r8/eval_zeroshot_lr1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Gemma_2_9b_it/phase1_medqa/evaluation/mcq/r8/eval_zeroshot_lr1e4_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_x86/bin/activate

export HF_HOME="/path/to/your/project_root/.hf_cache"

BASE_MODEL="./models/Gemma-2-9B-it"
CHECKPOINTS_DIR="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Gemma_2_9b_it/phase1_medqa/r8/lr_1e4"
OUTPUT_BASE="./fine_tuning/fine_tuning_sequential/vanilla/results/Gemma_2_9b_it/phase1_medqa/mcq/r8/lr_1e4/zeroshot"

CONFIGS="lr1.1247e-4_rank8 lr2.0000e-4_rank8 lr3.5566e-4_rank8 lr6.3246e-4_rank8"

for CONFIG in $CONFIGS; do
    ADAPTER_PATH="${CHECKPOINTS_DIR}/${CONFIG}/phase_1_final_adapter"
    OUTPUT_DIR="${OUTPUT_BASE}/${CONFIG}"
    MODEL_NAME="Vanilla_P1_${CONFIG}"

    mkdir -p "$OUTPUT_DIR"

    echo "=========================================================="
    echo " Zero-Shot | Config: ${CONFIG}"
    echo "=========================================================="

    echo "Evaluating PubMedQA..."
    python fine_tuning/eval_zeroshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "PubMedQA" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --dataset_path "./Data/MedicalQA/Data_Generation/PubMedQA/Test_Dev/pubmedqa_test.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "yes_no_maybe" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating MedQA..."
    python fine_tuning/eval_zeroshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "MedQA" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --dataset_path "./Data/MedicalQA/Data_Generation/MedQA/Test/medqa_test.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating MedMCQA..."
    python fine_tuning/eval_zeroshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "MedMCQA" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --dataset_path "./Data/MedicalQA/Data_Generation/MedMCQA/Test/medmcqa_test.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating Medicines..."
    python fine_tuning/eval_zeroshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "Medicines" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --dataset_path "./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/Finetuning_medicines_TEST_mcq.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating DDIs..."
    python fine_tuning/eval_zeroshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "DDIs" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --dataset_path "./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/final_TEST_mcq.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo ">>> ${CONFIG} done <<<"
done

echo "=================================================================="
echo " All Zero-Shot (lr_1e4) evaluations complete. Results in: ${OUTPUT_BASE}"
echo "=================================================================="
