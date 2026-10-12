#!/bin/bash
#SBATCH --job-name=Baseline_Gemma_calculate_metrics
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            
#SBATCH --time=00:30:00              


#SBATCH --output=baselines_evaluation/logs/Gemma-2-9B/calculate_metrics/calc_zeroshot_%j.out
#SBATCH --error=baselines_evaluation/logs/Gemma-2-9B/calculate_metrics/calc_zeroshot_%j.err

cd /path/to/your/project_root

module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

echo "=========================================="
echo "Starting Metrics Calculation Engine"
echo "=========================================="

TARGET_FOLDER="./baselines_evaluation/results/Gemma-2-9B/zeroshot"

python baselines_evaluation/calculate_metrics_new.py \
    --folder_path "$TARGET_FOLDER"

echo "=========================================="
echo "All metrics calculated!"
echo "=========================================="

