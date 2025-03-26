# Convolutional

## Model Description

Convolutional translation models leverage convolutional neural networks (CNNs) for machine translation tasks, offering
an alternative to traditional RNN-based approaches. These models process input sequences through multiple convolutional
layers, capturing local patterns and hierarchical features in the text. By using stacked convolutions with gated linear
units, they effectively model long-range dependencies while maintaining computational efficiency. Convolutional
translation models are particularly advantageous for parallel processing and handling large-scale translation tasks,
demonstrating competitive performance in sequence-to-sequence learning scenarios with reduced training time compared to
recurrent architectures.

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
bash prepare-wmt14en2de.sh
cd ../..

TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

### Install Dependencies

Convolutional model is using Fairseq toolbox. Before you run this model, you need to setup Fairseq first.

```bash
bash install_toolbox_fairseq.sh
```

## Model Training

```bash
# Train
mkdir -p checkpoints/fconv_wmt_en_de
fairseq-train data-bin/wmt17_en_de --arch fconv_wmt_en_de \
                                   --max-epoch 100 \
                                   --dropout 0.2 \
                                   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                                   --optimizer nag --clip-norm 0.1 \
                                   --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
                                   --max-tokens 4000 \
                                   --no-epoch-checkpoints \
                                   --save-dir checkpoints/fconv_wmt_en_de

# Evaluate
fairseq-generate data-bin/wmt17_en_de --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
                                      --beam 5 --remove-bpe
```

## Model Results

| Model         | GPUs       | QPS     | Train Epochs | Evaluate_Bleu |
|---------------|------------|---------|--------------|---------------|
| Convolutional | BI-V100 x8 | 1650.49 | 100          | 25.55         |

## References

- [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)
