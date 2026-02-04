#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate da3
module load gcc ffmpeg

scene_dir_path=$1

repo_path=/home/cizinsky/master-thesis
cd $repo_path/submodules/da3

python inference.py --scene-dir $scene_dir_path