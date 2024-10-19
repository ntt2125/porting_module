#!/bin/bash

export PYTHONPATH=PYTHONPATH:.

# torch
python2=/home/ntthong/miniconda3/envs/ecg/bin/python

#tensorflow
python1=/home/ntthong/miniconda3/envs/imlenet/bin/python



# $python1 porting_module/TF/conv1d.py
# $python2 porting_module/Torch/conv1d.py

# $python1 porting_module/TF/batchnorm.py
# $python2 porting_module/Torch/batchnorm.py

# $python1 porting_module/TF/bi_lstm.py
# $python2 porting_module/Torch/bi_lstm.py

# $python1 porting_module/TF/attention.py
# $python2 porting_module/Torch/attention.py

# $python1 porting_module/TF/residual.py
# $python2 porting_module/Torch/residual.py

# $python1 porting_module/TF/reshape.py
# $python2 porting_module/Torch/reshape.py

$python1 porting_module/TF/loss.py
$python2 porting_module/Torch/loss.py