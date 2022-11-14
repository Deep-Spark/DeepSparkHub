# BERT Pretraining

## Model description

BERT, or Bidirectional Encoder Representations from Transformers, improves upon standard Transformers by removing the unidirectionality constraint by using a masked language model (MLM) pre-training objective. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, BERT uses a next sentence prediction task that jointly pre-trains text-pair representations.

## Step 1: Installing

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip3 install -r requirements.txt
```

## Step 2: Download data

Download the [MNLI Dataset](http://www.nyu.edu/projects/bowman/multinli/)


## Step 3: Run BERT

```bash
# Make sure your dataset path is the same as above
bash train_bert.sh
```
