# BART

## Model Description

BART (Bidirectional and Auto-Regressive Transformer) is a powerful sequence-to-sequence model that combines
bidirectional and autoregressive approaches for natural language processing. Pretrained using a denoising objective,
BART excels in various tasks including text summarization, question answering, and dialogue generation. Its architecture
allows it to effectively handle both understanding and generation tasks, making it versatile for different NLP
applications. BART has demonstrated state-of-the-art performance on benchmarks like XSum, CNN/Daily Mail, and GLUE,
showcasing its robust capabilities in text transformation and comprehension.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

```bash
# Go to "toolbox/Fairseq" directory in root path
cd ../../../../toolbox/Fairseq/

# Download dataset
cd fairseq/
mkdir -p glue_data
cd glue_data/
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip
unzip RTE.zip
rm -rf RTE.zip

# Preprocess dataset
cd ..
./examples/roberta/preprocess_GLUE_tasks.sh glue_data RTE

# Download pretrain weight
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```

### Install Dependencies

Bart model is using Fairseq toolbox. Before you run this model, you need to setup Fairseq first.

```bash
bash install_toolbox_fairseq.sh
```

## Model Training

```bash
# Finetune on CLUE RTE task
bash bart.sh

# Inference on GLUE RTE task
python3 bart.py
```

## Model Results

| Model | GPUs       | QPS    | Train Epochs | Accuracy |
|-------|------------|--------|--------------|----------|
| BART  | BI-v100 x8 | 113.18 | 10           | 83.8     |

## References

- [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)
