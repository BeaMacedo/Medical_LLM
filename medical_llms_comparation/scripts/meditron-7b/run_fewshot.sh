#!/bin/bash
#SBATCH --job-name=meditron-7b_FS
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00

#SBATCH --output=medical_llms_comparation/logs/meditron-7b/fewshot/%x_%j.out
#SBATCH --error=medical_llms_comparation/logs/meditron-7b/fewshot/%x_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

OUTPUT_DIR="./medical_llms_comparation/results/meditron-7b/fewshot"
mkdir -p "$OUTPUT_DIR"
mkdir -p medical_llms_comparation/logs/meditron-7b/fewshot

echo "=========================================================="
echo " meditron-7b | Few-Shot | Dataset: MedQA"
echo "=========================================================="
python medical_llms_comparation/eval_fewshot_baseline.py \
    --model_name "Meditron-7b" \
    --dataset_name "MedQA" \
    --model_path "./models/Meditron-7B" \
    --dataset_path "./Data/MedicalQA/Data_Generation/MedQA/Test/medqa_test.jsonl" \
    --few_shot_path "./Data/MedicalQA/Data_Generation/MedQA/Test/few_shot_examples_medqa.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --answer_type "letters" \
    --max_length 4000

echo "=========================================================="
echo " meditron-7b | Few-Shot | Dataset: MedMCQA"
echo "=========================================================="
python medical_llms_comparation/eval_fewshot_baseline.py \
    --model_name "Meditron-7b" \
    --dataset_name "MedMCQA" \
    --model_path "./models/Meditron-7B" \
    --dataset_path "./Data/MedicalQA/Data_Generation/MedMCQA/Test/medmcqa_test.jsonl" \
    --few_shot_path "./Data/MedicalQA/Data_Generation/MedMCQA/Test/few_shot_examples_medmcqa.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --answer_type "letters" \
    --max_length 4000

echo "=========================================================="
echo " meditron-7b | Few-Shot | Dataset: PubMedQA"
echo "=========================================================="
python medical_llms_comparation/eval_fewshot_baseline.py \
    --model_name "Meditron-7b" \
    --dataset_name "PubMedQA" \
    --model_path "./models/Meditron-7B" \
    --dataset_path "./Data/MedicalQA/Data_Generation/PubMedQA/Test_Dev/pubmedqa_test.jsonl" \
    --few_shot_path "./Data/MedicalQA/Data_Generation/PubMedQA/Test_Dev/few_shot_examples_pubmedqa.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --answer_type "yes_no_maybe" \
    --max_length 4000

echo "=========================================================="
echo " meditron-7b | Few-Shot | Dataset: Medicines"
echo "=========================================================="
python medical_llms_comparation/eval_fewshot_baseline.py \
    --model_name "Meditron-7b" \
    --dataset_name "Medicines" \
    --model_path "./models/Meditron-7B" \
    --dataset_path "./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/Finetuning_medicines_TEST_mcq.jsonl" \
    --few_shot_path "./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/few_shot_examples_meds.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --answer_type "letters" \
    --max_length 4000

echo "=========================================================="
echo " meditron-7b | Few-Shot | Dataset: DDIs"
echo "=========================================================="
python medical_llms_comparation/eval_fewshot_baseline.py \
    --model_name "Meditron-7b" \
    --dataset_name "DDIs" \
    --model_path "./models/Meditron-7B" \
    --dataset_path "./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/final_TEST_mcq.jsonl" \
    --few_shot_path "./Data/Drugs/Data_Generation/DDIs/data_finetuning/final_datasets/few_shot_examples_ddi_complete.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --answer_type "letters" \
    --max_length 4000

echo "=================================================================="
echo " All Few-Shot datasets done for meditron-7b. Results in: $OUTPUT_DIR"
echo "=================================================================="
