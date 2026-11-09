#!/bin/bash
# FinBERT sentiment + BERTopic topics + call-level scores on the Unity cluster.
#   sbatch slurm/nlp.sbatch            (from the repo root, after the transcript pull)
# GPU makes FinBERT ~20x faster; on CPU it still works (~40 sentences/s).
# Sentiment is incremental, so a re-submitted job only scores new sentences.
#SBATCH --job-name=ecnlp-nlp
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda/latest 2>/dev/null || module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate ecnlp

# Same storage as slurm/pull_data.sbatch. Model weights are cached there too,
# keeping ~1 GB of Hugging Face downloads off the home quota.
export ECNLP_DATA_DIR="${ECNLP_DATA_DIR:-/project/pi_CHANGE_ME/earnings_nlp_data}"
export HF_HOME="${HF_HOME:-$ECNLP_DATA_DIR/hf_cache}"

nvidia-smi || true
python -m ecnlp.pipeline prepare
python -m ecnlp.pipeline nlp
