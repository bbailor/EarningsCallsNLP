#!/bin/bash
# NLP-03: run the data pull on the Unity cluster.
#   sbatch slurm/pull_data.sbatch            (from the repo root)
# CPU-only; the pull is network-bound. Long runs resume from cache if re-submitted.
#SBATCH --job-name=ecnlp-pull
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda/latest 2>/dev/null || module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate ecnlp

# Edit to your project storage path; keeps data off home quota.
export ECNLP_DATA_DIR="${ECNLP_DATA_DIR:-/project/pi_CHANGE_ME/earnings_nlp_data}"
# Key comes from .env in the repo root (chmod 600 .env) or the environment.

python -m ecnlp.pipeline check
python -m ecnlp.pipeline all
