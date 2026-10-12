#!/bin/bash
#SBATCH --job-name=TESTE_OpenQA_Qwen_TestSet
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

#SBATCH --output=baselines_evaluation/logs/Qwen2.5-7B-Instruct/%x_%j.out
#SBATCH --error=baselines_evaluation/logs/Qwen2.5-7B-Instruct/%x_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

OUTPUT_DIR="baselines_evaluation/results/Qwen2.5-7B-Instruct/long_form_evaluation/medicines_TESTE_alucinacoes"
mkdir -p "$OUTPUT_DIR"


TEST_MEDICINES="./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/teste.jsonl"


python baselines_evaluation/eval_open_ended.py \
    --model_name "Qwen2.5-7B-Instruct" \
    --model_path "./models/Qwen2.5-7B-Instruct" \
    --dataset_name "Medicines" \
    --dataset_path $TEST_MEDICINES \
    --output_dir $OUTPUT_DIR \
    --is_instruct
