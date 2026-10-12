#!/bin/bash
#SBATCH --job-name=Mixed_ExtMetrics_FS_1e4
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/evaluation/mcq/r16/extract_metrics_fewshot_lr1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/logs/Mixed_Baseline/evaluation/mcq/r16/extract_metrics_fewshot_lr1e4_%j.err

cd /path/to/your/project_root
source venv_x86/bin/activate

RESULTS_BASE="./fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/results/Mixed_Baseline/mcq/r16/lr_1e4/fewshot_2"
METRICS_DIR="${RESULTS_BASE}/metrics"
mkdir -p "$METRICS_DIR"

CONFIGS="lr1.1247e-4_rank16"

for CONFIG in $CONFIGS; do
    FOLDER="${RESULTS_BASE}/${CONFIG}"

    echo "=========================================================="
    echo " Config: ${CONFIG}"
    echo "=========================================================="

    echo "[1/2] Extracting answers..."
    python fine_tuning/extract_answers.py --folder_path "$FOLDER"

    echo "[2/2] Calculating metrics..."
    python fine_tuning/calculate_metrics_new.py \
        --folder_path "$FOLDER" \
        > "${METRICS_DIR}/${CONFIG}_metrics.txt"

    echo ">>> Metrics saved to: ${METRICS_DIR}/${CONFIG}_metrics.txt"
    echo ""
done

echo "=================================================================="
echo " All done! Metrics in: ${METRICS_DIR}"
echo "=================================================================="
