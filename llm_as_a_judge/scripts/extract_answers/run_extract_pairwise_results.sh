#!/bin/bash
#SBATCH --job-name=Judge_Extract_Pairwise_Scores
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00

#SBATCH --output=llm_as_a_judge/logs/pairwise/extraction/%x_%j.out
#SBATCH --error=llm_as_a_judge/logs/pairwise/extraction/%x_%j.err

cd /path/to/your/project_root
module load Python/3.12.3-GCCcore-13.3.0
source venv_x86/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python llm_as_a_judge/extract_pairwise_scores.py \
    --raw_dir "llm_as_a_judge/results"
