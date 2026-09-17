#!/bin/bash
#SBATCH --job-name=g2g_full
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/users/k2585234/emulator_seasonal/validation_g2g/logs/g2g_full_%j.out
#SBATCH --error=/scratch/users/k2585234/emulator_seasonal/validation_g2g/logs/g2g_full_%j.err

set -euo pipefail

ROOT=/scratch/users/k2585234/emulator_seasonal/validation_g2g
cd "$ROOT"

source /scratch/users/k2585234/emulator_seasonal/env/seasonal_emulator/bin/activate

python process_g2g_validation_full_hpc.py --root "$ROOT"
