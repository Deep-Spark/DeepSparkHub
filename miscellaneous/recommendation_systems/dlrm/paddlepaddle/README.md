# DLRM

## Model description
With the advent of deep learning, neural network-based recommendation models have emerged as an important tool for tackling personalization and recommendation tasks. These networks differ significantly from other deep learning networks due to their need to handle categorical features and are not well studied or understood. In this paper, we develop a state-of-the-art deep learning recommendation model (DLRM) and provide its implementation in both PyTorch and Caffe2 frameworks. In addition, we design a specialized parallelization scheme utilizing model parallelism on the embedding tables to mitigate memory constraints while exploiting data parallelism to scale-out compute from the fully-connected layers. We compare DLRM against existing recommendation models and characterize its performance on the Big Basin AI platform, demonstrating its usefulness as a benchmark for future algorithmic experimentation and system co-design.

## Step 1: Installation

```bash
git clone -b master --recursive https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec
git checkout eb869a15b7d858f9f3788d9b25af4f61a022f9c4
pip3 install -r requirements.txt

```

## Step 2: Preparing datasets

```bash
pushd datasets/criteo
sh run.sh
popd
```

## Step 3: Training


```bash
cd models/rank/dlrm
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
python3 -u ../../../tools/infer.py -m config_bigdata.yaml

```

## Results

| GPUs        | IPS       | AUC         |
|-------------|-----------|-------------|
| BI-V100 x1  | 300       | 0.802409    |

## Reference
- [DLRM](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.3.0/models/rank/dlrm)
