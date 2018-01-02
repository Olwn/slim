#!/usr/bin/env bash
set -e
# data
data_name=mnist
DATASET_DIR=/tmp/mnist
# model
model_name=lenet
# train
wd=0.001


python test.py \
      --dataset_name=${data_name} \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${model_name} \
      --batch_size=50