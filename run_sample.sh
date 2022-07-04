#!/bin/sh 
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate baseline_env/

python train.py --config configs/train/sample.yaml --gpus 0
