# Convolutional

## Model description
The following instructions can be used to train a Convolutional translation model on the WMT English to German dataset. 

## Envoronment
Before you run this model, please refer to [README_environment.md](README_environment.md) to setup Fairseq and install requirements.

## Download and preprocess data

```
cd fairseq/examples/translation/
bash prepare-wmt14en2de.sh
cd ../..

TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

## Train

```
mkdir -p checkpoints/fconv_wmt_en_de
fairseq-train \
    data-bin/wmt17_en_de \
    --arch fconv_wmt_en_de \
    --max-epoch 100 \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --no-epoch-checkpoints \
    --save-dir checkpoints/fconv_wmt_en_de
```

## Evaluate

```
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

## Results on BI-V100

```
| GPUs | QPS | Train Epochs | Evaluate_Bleu  |
|------|-----|--------------|------------|
| 1x8  | 1650.49 | 100           | 25.55 |