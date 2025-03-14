# Transformer

## Model description

The following instructions can be used to train a Transformer model on the IWSLT'14 German to English dataset.

## Step 1: Installation

Transformer model is using Fairseq toolbox. Before you run this model, you need to setup Fairseq first.

```bash
# Go to "toolbox/Fairseq" directory in root path
cd ../../../../toolbox/Fairseq/
bash install_toolbox_fairseq.sh
```

## Step 2: Preparing datasets

```bash
cd fairseq/examples/translation/
bash prepare-iwslt14.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

## Step 3: Training

```bash
# Train
mkdir -p checkpoints/transformer
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 100 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer \
    --no-epoch-checkpoints \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# Evaluate
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/transformer/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```

## Results

| GPUs       | QPS     | Train Epochs | Bleu  |
|------------|---------|--------------|-------|
| BI-v100 x8 | 3204.78 | 100          | 35.07 |

## Reference

- [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)
