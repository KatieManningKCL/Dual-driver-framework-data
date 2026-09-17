#!/bin/bash
#SBATCH --job-name=cds_val
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/k2585234/emulator_seasonal/validation_cds_eurocordex/logs/cds_val_%j.out
#SBATCH --error=/scratch/users/k2585234/emulator_seasonal/validation_cds_eurocordex/logs/cds_val_%j.err

set -euo pipefail

cd /scratch/users/k2585234/emulator_seasonal
source env/seasonal_emulator/bin/activate

mkdir -p /scratch/users/k2585234/emulator_seasonal/validation_cds_eurocordex/logs
mkdir -p /scratch/users/k2585234/emulator_seasonal/validation_cds_eurocordex/outputs

python /scratch/users/k2585234/emulator_seasonal/validation_cds_eurocordex/scripts/run_cds_eurocordex_validation.py

