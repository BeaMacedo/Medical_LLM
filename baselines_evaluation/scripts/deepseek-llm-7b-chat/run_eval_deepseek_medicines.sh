#!/bin/bash
#SBATCH --job-name=DeepSeek_Medicines_zs
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

#SBATCH --output=baselines_evaluation/logs/deepseek-llm-7b-chat/zeroshot/%x_%j.out
#SBATCH --error=baselines_evaluation/logs/deepseek-llm-7b-chat/zeroshot/%x_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

mkdir -p baselines_evaluation/results/deepseek-llm-7b-chat
mkdir -p baselines_evaluation/logs/deepseek-llm-7b-chat/zeroshot

python baselines_evaluation/eval_engine.py \
    --model_name "deepseek-llm-7b-chat" \
    --dataset_name "Medicines" \
    --model_path "./models/deepseek-llm-7b-chat" \
    --dataset_path "./Data/Drugs/Data_Generation/Medicines/data_finetuning/final_datasets/Finetuning_medicines_TEST_mcq.jsonl" \
    --output_dir "./baselines_evaluation/results/deepseek-llm-7b-chat/zeroshot" \
    --answer_type "letters" \
    --max_length 8000 \
    --is_instruct
