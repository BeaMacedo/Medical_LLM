#!/bin/bash
#SBATCH --job-name=V_P2_FS_1e5
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/evaluation/mcq/r32/eval_fewshot_lr1e5_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/evaluation/mcq/r32/eval_fewshot_lr1e5_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_x86/bin/activate

export HF_HOME="/path/to/your/project_root/.hf_cache"

BASE_MODEL="./models/Qwen2.5-7B-Instruct"
PREV_ADAPTER="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase1_medqa/r8/lr_1e4/lr6.3246e-4_rank8/phase_1_final_adapter"
CHECKPOINTS_DIR="./fine_tuning/fine_tuning_sequential/vanilla/checkpoints/Qwen2.5-7B-Instruct/phase2_medicines/r32/lr_1e5"
OUTPUT_BASE="./fine_tuning/fine_tuning_sequential/vanilla/results/Qwen2.5-7B-Instruct/phase2_medicines/mcq/r32/lr_1e5/fewshot"


CONFIGS="lr3.5566e-5_rank32"

for CONFIG in $CONFIGS; do
    ADAPTER_PATH="${CHECKPOINTS_DIR}/${CONFIG}/phase_2_final_adapter"
    OUTPUT_DIR="${OUTPUT_BASE}/${CONFIG}"
    MODEL_NAME="Vanilla_P2_${CONFIG}"

    mkdir -p "$OUTPUT_DIR"

    echo "=========================================================="
    echo " Few-Shot | Config: ${CONFIG}"
    echo "=========================================================="

    echo "Evaluating PubMedQA..."
    python fine_tuning/eval_fewshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "PubMedQA" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --prev_adapters_paths "$PREV_ADAPTER" \
        --dataset_path "./Data/MedicalQA/Data_Generation/PubMedQA/Test_Dev/pubmedqa_test.jsonl" \
        --few_shot_path "./Data/MedicalQA/Data_Generation/PubMedQA/Test_Dev/few_shot_examples_pubmedqa.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "yes_no_maybe" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating MedQA..."
    python fine_tuning/eval_fewshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "MedQA" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --prev_adapters_paths "$PREV_ADAPTER" \
        --dataset_path "./Data/MedicalQA/Data_Generation/MedQA/Test/medqa_test.jsonl" \
        --few_shot_path "./Data/MedicalQA/Data_Generation/MedQA/Test/few_shot_examples_medqa.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating MedMCQA..."
    python fine_tuning/eval_fewshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "MedMCQA" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --prev_adapters_paths "$PREV_ADAPTER" \
        --dataset_path "./Data/MedicalQA/Data_Generation/MedMCQA/Test/medmcqa_test.jsonl" \
        --few_shot_path "./Data/MedicalQA/Data_Generation/MedMCQA/Test/few_shot_examples_medmcqa.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating Medicines..."
    python fine_tuning/eval_fewshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "Medicines" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --prev_adapters_paths "$PREV_ADAPTER" \
        --dataset_path "./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/Finetuning_medicines_TEST_mcq.jsonl" \
        --few_shot_path "./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/few_shot_examples_meds.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo "Evaluating DDIs..."
    python fine_tuning/eval_fewshot_phase_new.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "DDIs" \
        --model_path "$BASE_MODEL" \
        --peft_model_path "$ADAPTER_PATH" \
        --prev_adapters_paths "$PREV_ADAPTER" \
        --dataset_path "./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/final_TEST_mcq.jsonl" \
        --few_shot_path "./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/few_shot_examples_ddi_complete.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --answer_type "letters" \
        --max_length 8000 \
        --is_instruct

    echo ">>> ${CONFIG} done <<<"
done

echo "=================================================================="
echo " All Few-Shot (lr_1e5) evaluations complete. Results in: ${OUTPUT_BASE}"
echo "=================================================================="
