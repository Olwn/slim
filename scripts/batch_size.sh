#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset
# 2. Trains a LeNet model on the MNIST training set.
# 3. Evaluates the model on the MNIST testing set.
#
# Usage:
# cd slim
# ./slim/scripts/train_lenet_on_mnist.sh
set -e
home=`env | grep ^HOME= | cut -c 6-`
# gpu
gpu_tr=$1
gpu_ts=
# model
model_name=lenet
data_name=cifar10
preprocess=custom
image_size=32
# training
ts_batch_size=500
lr=0.01
lr_policy=exponential
# log
save_interval=10
t=`date +%m%d%H%M%S`

# Where the dataset is saved to.
DATASET_DIR=/tmp/${data_name}

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=${data_name} \
  --dataset_dir=${DATASET_DIR}

# Run training.
for bs in 16 32 64 128 256 512 1024 2048;
do
TRAIN_DIR="${home}/exp/slim/bs${bs}-${data_name}-${model_name}-lr${lr}-${t}"
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${data_name} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --preprocessing_name=${preprocess} \
  --train_image_size=${image_size} \
  --max_number_of_steps=100000 \
  --decay_steps=2000 \
  --batch_size=${bs} \
  --learning_rate=${lr} \
  --save_interval_secs=${save_interval} \
  --save_summaries_secs=${save_interval} \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate_decay_type=${lr_policy} \
  --learning_rate_decay_factor=0.91 \
  --weight_decay=0.0 \
  --noisy_gradient 0 \
  --clone_on_cpu=0 \
  --gpu=${gpu_tr} \
  --num_clones=2 &

# Run evaluation.

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --batch_size=${ts_batch_size} \
  --dataset_name=${data_name} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --eval_image_size=${image_size} \
  --timeout=`expr ${save_interval} + 5` \
  --model_name=${model_name} \
  --preprocessing_name=${preprocess} \
  --gpu=${gpu_ts}
done