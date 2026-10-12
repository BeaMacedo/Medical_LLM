#!/bin/bash
#SBATCH --job-name=V_P2_ExtMetrics_FS_1e5
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/evaluation/mcq/r32/extract_metrics_fewshot_lr1e5_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Qwen2.5-7B-Instruct/phase2_medicines/evaluation/mcq/r32/extract_metrics_fewshot_lr1e5_%j.err

cd /path/to/your/project_root
source venv_x86/bin/activate

RESULTS_BASE="./fine_tuning/fine_tuning_sequential/vanilla/results/Qwen2.5-7B-Instruct/phase2_medicines/mcq/r32/lr_1e5/fewshot"
METRICS_DIR="${RESULTS_BASE}/metrics"
mkdir -p "$METRICS_DIR"

CONFIGS="lr3.5566e-5_rank32"

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
