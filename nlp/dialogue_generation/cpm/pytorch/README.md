# CPM

## Model description

Pre-trained Language Models (PLMs) have proven to be beneficial for various downstream NLP tasks. Recently, GPT-3, with 175 billion parameters and 570GB training data, drew a lot of attention due to the capacity of few-shot (even zero-shot) learning. However, applying GPT-3 to address Chinese NLP tasks is still challenging, as the training corpus of GPT-3 is primarily English, and the parameters are not publicly available. In this technical report, we release the Chinese Pre-trained Language Model (CPM) with generative pre-training on large-scale Chinese training data. To the best of our knowledge, CPM, with 2.6 billion parameters and 100GB Chinese training data, is the largest Chinese pre-trained language model, which could facilitate several downstream Chinese NLP tasks, such as conversation, essay generation, cloze test, and language understanding. Extensive experiments demonstrate that CPM achieves strong performance on many NLP tasks in the settings of few-shot (even zero-shot) learning. The code and parameters are available at https://github.com/TsinghuaAI/CPM-Generate.

## Step 1: Installing packages

### Install packages in container

```shell
$ apt install -y numactl
or 
$ yum install -y numactl
```

## Step 2:  Download dataset and model

```shell
$ pip install gdown
$ mkdir -p /home/data/perf/cpm
$ cd /home/data/perf/cpm
$ gdown -O "STC.json" --fuzzy https://drive.google.com/uc?id=19VyP6e7pS4pYed87yfvO2hAO7dd0dL9K
```

Download pretrained model file "model-v2.tar.gz": https://wudaoai.cn/model/download?resourceId=1420992356135514112&filename=CPM-1-2.6B-zh.tar.gz

```shell
$ tar -xvf model-v2.tar.gz
```



## Step 3: Training

### On single GPU

```shell
$ cd modelzoo-benchmark/nlp/dialogue_generation/cpm/pytorch/base/
$ python3  prepare.py --name iluvatar --data_dir /home/data/perf/cpm
$ bash run_training.sh --name iluvatar --config V100x1x8 --data_dir /home/data/perf/cpm
```

### On single GPU (AMP)

```shell
$ bash run_training.sh --name iluvatar --config V100x1x1 --data_dir /home/data/perf/cpm
```

## Results on BI-V100

| GPUs | FPS | E2E      | Accuracy |
|------| --- |----------| -------- |
| 1x8  | 152.86 | 3558.36s | 0.91     |


| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 0.91                 | SDK V2.2,bs:128,8x,AMP                   | 357         | 0.91     | 156\*8     | 0.93        | 20.6\*8                 | 1         |



## Reference
https://github.com/TsinghuaAI/CPM-1-Finetune