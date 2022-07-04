#!/bin/sh 
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
rm -rf baseline_env/
conda create --prefix baseline_env/ python=3.7 -y
conda activate baseline_env/

pip install -r requirements.txt

