#!/bin/bash
#SBATCH --job-name=Q_B_calculate_metrics
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            
#SBATCH --time=00:30:00              


#SBATCH --output=baselines_evaluation/logs/Qwen2.5-7B-Instruct/calculate_metrics/calc_fewshot_%j.out
#SBATCH --error=baselines_evaluation/logs/Qwen2.5-7B-Instruct/calculate_metrics/calc_fewshot_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

echo "=========================================="
echo "Starting Metrics Calculation Engine"
echo "=========================================="

TARGET_FOLDER="./baselines_evaluation/results/Qwen2.5-7B-Instruct/fewshot"

python baselines_evaluation/calculate_metrics_new.py \
    --folder_path "$TARGET_FOLDER"

echo "=========================================="
echo "All metrics calculated!"
echo "=========================================="

