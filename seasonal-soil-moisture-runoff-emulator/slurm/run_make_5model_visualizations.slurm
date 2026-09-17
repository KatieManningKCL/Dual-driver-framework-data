#!/bin/bash -l
#SBATCH --job-name=emu5_png
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/k2585234/emulator_seasonal/logs_v2_5models/emu5_png_%j.out
#SBATCH --error=/scratch/users/k2585234/emulator_seasonal/logs_v2_5models/emu5_png_%j.err

set -euo pipefail

ROOT=/scratch/users/k2585234/emulator_seasonal
CODE_DIR="${ROOT}/scripts/HPC_code_v2_5models"
mkdir -p "${ROOT}/logs_v2_5models"

source "${ROOT}/env/seasonal_emulator/bin/activate"
cd "${CODE_DIR}"

python make_5model_visualizations.py --root "${ROOT}"
