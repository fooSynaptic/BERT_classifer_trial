export BERT_BASE_DIR=/data2/sda/jiaxin_hu/bert/pre_trained_model/chinese_L-12_H-768_A-12
export MY_DATASET=./data
export OUT_DIR=./out_dir/

python3.5 run_classifier.py \
  --task_name=selfsim \
  --do_train=false \
  --do_predict=true \
  --do_eval=false \
  --dopredict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \ #模型参数
