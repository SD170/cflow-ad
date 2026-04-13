#!/bin/bash
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -t 0-04:00:00
#SBATCH -p general
#SBATCH -A class_cse579spring2026
#SBATCH -q class
#SBATCH -o logs/cflow-infer-%j.out
#SBATCH -e logs/slurm-infer.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="sdutta60@asu.edu"
#SBATCH --export=NONE

# --- edit these per run ---
# Must match training: dataset, class, -inp, encoder, run name inside checkpoint filename.
export DATASET=btad
export CLASS_NAME=01
export INP=512
# Path relative to repo root OR absolute path to .pt file:
export CKPT="weights/REPLACE_WITH_YOUR_CHECKPOINT.pt"
export CSV_OUT="inference_outputs/${CLASS_NAME}_scores.csv"
export HEATMAP_DIR="inference_outputs/heatmaps/${CLASS_NAME}"

echo "Starting inference export..."
cd "${SCRATCH}/cflow-ad" || exit 1
mkdir -p logs inference_outputs/heatmaps

module load mamba/latest
source activate pytorch-gpu-2.3.1-cuda-12.1

python -m pip install -q -U pip
python -m pip install -q -r requirements.txt
python -m pip install -q "git+https://github.com/vislearn/FrEIA.git@cc5cf5ebee08f9bb762bab5a6535c11d19ccb026"

python scripts/infer_export.py \
  --dataset "${DATASET}" \
  --class-name "${CLASS_NAME}" \
  -inp "${INP}" \
  --checkpoint "${CKPT}" \
  --csv-out "${CSV_OUT}" \
  --heatmap-dir "${HEATMAP_DIR}" \
  --gpu 0

echo "Done $(date)"
