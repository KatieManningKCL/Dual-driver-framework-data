#!/bin/bash -l
#SBATCH --job-name=emu5_allrcp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/k2585234/emulator_seasonal/logs_v2_5models/emu5_allrcp_%j.out
#SBATCH --error=/scratch/users/k2585234/emulator_seasonal/logs_v2_5models/emu5_allrcp_%j.err

set -euo pipefail

ROOT=/scratch/users/k2585234/emulator_seasonal
CODE_DIR="${ROOT}/scripts/HPC_code_v2_5models"
mkdir -p "${ROOT}/logs_v2_5models"

source "${ROOT}/env/seasonal_emulator/bin/activate"
cd "${CODE_DIR}"

python make_ppt_figures_all_rcps_by_model.py --root "${ROOT}"

cd "${ROOT}"
tar -czf outputs_v2_5models_ppt_figures_all_rcps_by_model.tar.gz outputs_v2_5models/ppt_figures_all_rcps_by_model
ls -lh outputs_v2_5models_ppt_figures_all_rcps_by_model.tar.gz
