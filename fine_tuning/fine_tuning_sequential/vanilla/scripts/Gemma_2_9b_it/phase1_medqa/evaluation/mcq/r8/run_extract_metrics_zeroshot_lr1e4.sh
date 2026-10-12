#!/bin/bash
#SBATCH --job-name=V_P1_ExtMetrics_ZS_1e4
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=fine_tuning/fine_tuning_sequential/vanilla/logs/Gemma_2_9b_it/phase1_medqa/evaluation/mcq/r8/extract_metrics_zeroshot_lr1e4_%j.out
#SBATCH --error=fine_tuning/fine_tuning_sequential/vanilla/logs/Gemma_2_9b_it/phase1_medqa/evaluation/mcq/r8/extract_metrics_zeroshot_lr1e4_%j.err

cd /path/to/your/project_root
source venv_x86/bin/activate

RESULTS_BASE="./fine_tuning/fine_tuning_sequential/vanilla/results/Gemma_2_9b_it/phase1_medqa/mcq/r8/lr_1e4/zeroshot"
METRICS_DIR="${RESULTS_BASE}/metrics"
mkdir -p "$METRICS_DIR"

CONFIGS="lr1.1247e-4_rank8 lr2.0000e-4_rank8 lr3.5566e-4_rank8 lr6.3246e-4_rank8"

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
