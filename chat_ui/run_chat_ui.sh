#!/bin/bash
#SBATCH --job-name=ChatUI
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=chat_ui/logs/chat_ui_%j.out
#SBATCH --error=chat_ui/logs/chat_ui_%j.err

cd /path/to/your/project_root
mkdir -p chat_ui/logs

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA
source venv_chatui/bin/activate

export HF_HOME="/path/to/your/project_root/.hf_cache"

PORT=7860

echo "=================================================================="
echo " Chat UI job started"
echo " Node (use this hostname to the SSH tunnel): $(hostname)"
echo " Port: ${PORT}"
echo "=================================================================="

python chat_ui/chat_app.py --server_port ${PORT}
