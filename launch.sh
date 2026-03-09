#!/bin/bash
# Launch script for BERT classifier training/prediction
# 
# IMPORTANT: Update the paths below to match your environment before running

# Configuration
export BERT_BASE_DIR=/data2/sda/jiajia/bert/pre_trained_model/chinese_L-12_H-768_A-12
export MY_DATASET=./dataset
export OUT_DIR=./output

# Create output directory if it doesn't exist
mkdir -p $OUT_DIR

# Run BERT classifier
# Mode can be changed by modifying the flags below:
# --do_train=true|false
# --do_eval=true|false
# --do_predict=true|false
python3 run_classifier.py \
  --task_name=selfsim \
  --do_train=false \
  --do_predict=true \
  --do_eval=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=$OUT_DIR \
  "$@"
