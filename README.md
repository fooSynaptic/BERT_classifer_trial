# BERT Classifier Trial

A BERT-based text classifier for Chinese corpus classification tasks.

## Description

This project implements a BERT (Bidirectional Encoder Representations from Transformers) 
classifier for Chinese text classification. It's built on top of Google's BERT implementation
and supports fine-tuning for custom classification tasks.

## Features

- BERT-based Chinese text classification
- Support for custom datasets
- Training, evaluation, and inference modes
- TPU/GPU/CPU support

## Prerequisites

- Python 3.5+
- TensorFlow >= 1.11.0
- Pre-trained Chinese BERT model

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd BERT_classifer_trial
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained Chinese BERT model:
```bash
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
```

## Project Structure

```
BERT_classifer_trial/
├── modeling.py           # BERT model implementation
├── optimization.py       # Optimizer with weight decay
├── tokenization.py       # Text tokenization utilities
├── run_classifier.py     # Main training/inference script
├── predict_eval.py       # Evaluation script for predictions
├── launch.sh             # Launch script for training/prediction
├── requirements.txt      # Python dependencies
└── dataset/              # Dataset directory
    ├── train.csv
    ├── dev.csv
    └── test.csv
```

## Dataset Format

The dataset should be in CSV format with `<>` as the delimiter:

```
label<>text_a<>text_b
```

Example:
```
1<>这是一段示例文本<>另一段文本
0<>另一个例子<>对应文本
```

## Usage

### Training

1. Update `launch.sh` with your paths:
```bash
export BERT_BASE_DIR=/path/to/chinese_L-12_H-768_A-12
export MY_DATASET=./dataset
export OUT_DIR=./output
```

2. Run training:
```bash
bash launch.sh
```

Or run directly:
```bash
python run_classifier.py \
  --task_name=selfsim \
  --do_train=true \
  --do_eval=true \
  --data_dir=./dataset \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./output
```

### Inference

To run inference on test data:
```bash
python run_classifier.py \
  --task_name=selfsim \
  --do_predict=true \
  --data_dir=./dataset \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=./output/model.ckpt \
  --max_seq_length=128 \
  --output_dir=./output
```

### Evaluation

After inference, evaluate the results:
```bash
python predict_eval.py
```

## Configuration

Key parameters in `run_classifier.py`:

- `--task_name`: Task name (selfsim, cola, mnli, mrpc, xnli)
- `--do_train`: Enable training mode
- `--do_eval`: Enable evaluation mode
- `--do_predict`: Enable prediction mode
- `--max_seq_length`: Maximum sequence length (default: 128)
- `--train_batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs (default: 3.0)

## Supported Tasks

- `selfsim`: Custom similarity classification (binary: 0, 1)
- `cola`: Corpus of Linguistic Acceptability
- `mnli`: Multi-Genre Natural Language Inference
- `mrpc`: Microsoft Research Paraphrase Corpus
- `xnli`: Cross-lingual Natural Language Inference

## License

Apache License 2.0 (following Google's BERT)

## Acknowledgments

- Google's BERT: https://github.com/google-research/bert
- Pre-trained Chinese model from Google
