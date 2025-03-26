# CPM

## Model Description

CPM (Chinese Pre-trained Language Model) is a large-scale generative language model specifically designed for Chinese
NLP tasks. With 2.6 billion parameters trained on 100GB of Chinese text data, CPM demonstrates exceptional performance
in various applications including conversation generation, essay writing, and language understanding. Its architecture
enables effective few-shot and zero-shot learning capabilities, making it particularly valuable for Chinese language
processing. As one of the largest Chinese language models, CPM significantly advances the state of Chinese NLP research
and applications.s

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```shell
pip install gdown
mkdir -p /home/data/perf/cpm
cd /home/data/perf/cpm
gdown -O "STC.json" --fuzzy https://drive.google.com/uc?id=19VyP6e7pS4pYed87yfvO2hAO7dd0dL9K
```

Download pretrained model file "model-v2.tar.gz":
<https://wudaoai.cn/model/download?resourceId=1420992356135514112&filename=CPM-1-2.6B-zh.tar.gz>

```shell
tar -xvf model-v2.tar.gz
```

### Install Dependencies

```shell
# Ubuntu
apt install -y numactl
# CentOS
yum install -y numactl
```

## Model Training

```shell
# On single GPU
cd modelzoo-benchmark/nlp/dialogue_generation/cpm/pytorch/base/
python3  prepare.py --name iluvatar --data_dir /home/data/perf/cpm
bash run_training.sh --name iluvatar --config V100x1x8 --data_dir /home/data/perf/cpm

# On single GPU (AMP)
bash run_training.sh --name iluvatar --config V100x1x1 --data_dir /home/data/perf/cpm
```

## Model Results

| Model | GPUs       | FPS    | E2E      | Accuracy |
|-------|------------|--------|----------|----------|
| CPM   | BI-V100 x8 | 152.86 | 3558.36s | 0.91     |

| Convergence criteria | Configuration          | Performance | Accuracy | Power (W) | Scalability | Memory utilization (G) | Stability |
|----------------------|------------------------|-------------|----------|-----------|-------------|------------------------|-----------|
| 0.91                 | SDK V2.2,bs:128,8x,AMP | 357         | 0.91     | 156\*8    | 0.93        | 20.6\*8                | 1         |

## References

- [CPM-1-Finetune](https://github.com/TsinghuaAI/CPM-1-Finetune)
