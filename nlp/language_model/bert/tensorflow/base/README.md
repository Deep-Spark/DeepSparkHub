# BERT Pretraining

## Model description
BERT, or Bidirectional Encoder Representations from Transformers, improves upon standard Transformers by removing the unidirectionality constraint by using a masked language model (MLM) pre-training objective. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, BERT uses a next sentence prediction task that jointly pre-trains text-pair representations.

## Prepare

### Install packages

```shell
bash init_tf.sh
```

### Download datasets

This [Google Drive location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) contains the following.  
You need to download tf1_ckpt folde , vocab.txt and bert_config.json into one file named bert_pretrain_ckpt_tf

```
bert_pretrain_ckpt_tf: contains checkpoint files
    model.ckpt-28252.data-00000-of-00001
    model.ckpt-28252.index
    model.ckpt-28252.meta
    vocab.txt
    bert_config.json
```
[Download and preprocess datasets](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#generate-the-tfrecords-for-wiki-dataset)
You need to make a file named  bert_pretrain_tf_records and store the results above.
tips: you can git clone this repo in other place ,we need the bert_pretrain_tf_records results here.


## Training

### Training on single card

```shell
bash run_1card_FPS.sh
```

### Training on mutil-cards
```shell
bash run_multi_card_FPS.sh 
```
 
## Result

|               | acc       |       fps |
| ---           | ---       | ---       |
|    multi_card |  0.424126  | 0.267241|
