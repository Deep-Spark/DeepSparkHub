# Transformer

## Model Description

The Transformer model revolutionized natural language processing with its attention-based architecture, eliminating the
need for recurrent connections. It employs self-attention mechanisms to process input sequences in parallel, capturing
long-range dependencies more effectively than previous models. Transformers excel in tasks like translation, text
generation, and summarization by dynamically weighting the importance of different words in a sequence. Their parallel
processing capability enables faster training and better scalability, making them the foundation for state-of-the-art
language models like BERT and GPT.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

```bash
# Go to "toolbox/Fairseq" directory in root path
cd ../../../../toolbox/Fairseq/

cd fairseq/examples/translation/
bash prepare-iwslt14.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### Install Dependencies

Transformer model is using Fairseq toolbox. Before you run this model, you need to setup Fairseq first.

```bash
bash install_toolbox_fairseq.sh
```

## Model Training

```bash
# Train
mkdir -p checkpoints/transformer
fairseq-train data-bin/iwslt14.tokenized.de-en \
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

## Model Results

| Model       | GPUs       | QPS     | Train Epochs | Bleu  |
|-------------|------------|---------|--------------|-------|--|
| Transformer | BI-V100 x8 | 3204.78 | 100          | 35.07 |

## References

- [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)
