#!/bin/bash
set -euo pipefail

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate da3
module load gcc ffmpeg

seq_name=$1
output_dir=/scratch/izar/cizinsky/thesis/results/$seq_name
mkdir -p $output_dir
cd submodules/da3
python inference.py --output_dir $output_dir