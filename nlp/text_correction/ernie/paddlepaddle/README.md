# ERNIE

## Introduction

ERNIE (Enhanced Representation through Knowledge Integration) is a family of large-scale pre-trained language models
developed by Baidu. ERNIE is designed to improve on traditional language models by incorporating knowledge from various
sources, such as structured knowledge graphs, and by integrating multiple linguistic features, including syntax,
semantics, and common sense. The model achieves this by using a knowledge-enhanced pre-training approach, which helps
ERNIE better understand and generate more accurate and contextually aware language representations.

## Step 1: Installing

```sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

```sh
cd PaddleNLP
pip3 install -r requirements.txt
```

## Step 2: Training

```sh
cd examples/text_correction/ernie-csc
pip3 install -r requirements.txt


python3 download.py --data_dir ./extra_train_ds/ --url https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml
python3 change_sgml_to_txt.py -i extra_train_ds/train.sgml -o extra_train_ds/train.txt

export CUDA_VISIBLE_DEVICES=0
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
python3 -u train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5 --model_name_or_path ernie-1.0 --output_dir ./checkpoints/ --extra_train_ds_dir ./extra_train_ds/ --max_seq_length 192
```

## Reference

- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
