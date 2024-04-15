#!/bin/bash
eval "$(conda shell.bash hook)"
export CONDA_ALWAYS_YES="true"
if [ -f environment.yml ]; then
  conda env create -f environment.yml
else
  conda create -n torch-stress python=3.11
  conda activate torch-stress
  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
  mkdir pip-build
  TMPDIR=pip-build pip --no-input --no-cache-dir install nvidia-ml-py==11.515.75 urwid==2.1.2 texttable==1.6.4
  rm -rf pip-build
  conda env export | grep -v "^prefix: " > environment.yml
fi
