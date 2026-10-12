#!/bin/bash
#SBATCH --job-name=meditron-7b_ExtMetrics_FS
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

#SBATCH --output=medical_llms_comparation/logs/meditron-7b/fewshot/extract_metrics_%j.out
#SBATCH --error=medical_llms_comparation/logs/meditron-7b/fewshot/extract_metrics_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

RESULTS_DIR="./medical_llms_comparation/results/meditron-7b/fewshot"
METRICS_DIR="${RESULTS_DIR}/metrics"
mkdir -p "$METRICS_DIR"

echo "=========================================================="
echo " meditron-7b — Few-Shot Extraction + Metrics"
echo "=========================================================="

echo "[1/2] Extracting answers..."
python fine_tuning/extract_answers.py --folder_path "$RESULTS_DIR"

echo "[2/2] Calculating metrics..."
python fine_tuning/calculate_metrics_new.py \
    --folder_path "$RESULTS_DIR" \
    > "${METRICS_DIR}/meditron-7b_fewshot_metrics.txt"

echo "Metrics saved to: ${METRICS_DIR}/meditron-7b_fewshot_metrics.txt"
