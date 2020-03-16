#!/bin/bash

devices=$1
save_name=$2

CUDA_VISIBLE_DEVICES=$devices python -u train.py \
--seed 90 \
--save ${save_name} \
--batch_size 3
