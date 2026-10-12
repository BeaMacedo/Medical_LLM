#!/bin/bash
#SBATCH --job-name=Judge_Plot_Pairwise_Mixed_vs_Baseline
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00

#SBATCH --output=llm_as_a_judge/logs/pairwise/plots/%x_%j.out
#SBATCH --error=llm_as_a_judge/logs/pairwise/plots/%x_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RESULTS_DIR="llm_as_a_judge/results"
ANALYSIS_DIR="llm_as_a_judge/analysis/pairwise_criteria/Mixed_lr1.1247e-4_rank16_vs_Qwen2.5-7B-Instruct"

JUDGE_MODELS=("Gemma-3-27B-PT" "Llama-3.3-70B-Instruct")
RUN_TYPE="pairwise_criteria"
PAIR_DIR="Mixed_lr1.1247e-4_rank16_vs_Qwen2.5-7B-Instruct"
DATASETS=("Medicines" "DDIs")

for judge in "${JUDGE_MODELS[@]}"; do
    result_files=()
    for dataset in "${DATASETS[@]}"; do
        f="$RESULTS_DIR/$judge/$RUN_TYPE/$PAIR_DIR/${dataset}_judge_pairwise_evaluation.json"
        if [[ -f "$f" ]]; then
            result_files+=("$f")
        else
            echo "[SKIP] Missing extracted results for judge=$judge dataset=$dataset (run run_judge_*.sh + run_extract_pairwise_results.sh first)"
        fi
    done

    if [[ "${#result_files[@]}" -gt 0 ]]; then
        python llm_as_a_judge/plot_pairwise_results.py \
            --results_paths "${result_files[@]}" \
            --output_dir "$ANALYSIS_DIR/$judge"
    fi
done