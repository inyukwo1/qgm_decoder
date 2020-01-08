#!/bin/bash

devices=$1
save_name=$2

CUDA_VISIBLE_DEVICES=$devices python -u ours/eval.py \
--seed 90 \
--save ${save_name} \
--batch_size 6 \
--load_model ./saved_model/semql_0.6531.model

#python sem2SQL.py --data_path ./data --input_path predict_lf.json --output_path ${save_name}

