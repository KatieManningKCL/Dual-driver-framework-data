#!/bin/bash -l
#SBATCH --job-name=emu5_models
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/scratch/users/k2585234/emulator_seasonal/logs_v2_5models/emu5_%A_%a.out
#SBATCH --error=/scratch/users/k2585234/emulator_seasonal/logs_v2_5models/emu5_%A_%a.err

set -euo pipefail

ROOT=/scratch/users/k2585234/emulator_seasonal
CODE_DIR="${ROOT}/scripts/HPC_code_v2_5models"
mkdir -p "${ROOT}/logs_v2_5models" "${ROOT}/outputs_v2_5models"

source "${ROOT}/env/seasonal_emulator/bin/activate"
cd "${CODE_DIR}"

MODELS=(elastic_net forest lightgbm process_hybrid mlp_stack)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Starting model: ${MODEL}"
echo "Host: $(hostname)"
echo "Start time: $(date)"

python run_model_all_scenarios.py \
  --root "${ROOT}" \
  --model "${MODEL}" \
  --train-rows 1200000 \
  --val-rows 300000 \
  --predict-chunk 50000 \
  --threads "${SLURM_CPUS_PER_TASK}"

echo "Finished model: ${MODEL}"
echo "End time: $(date)"
