# DeepFM

## Description

DeepFM (Deep Factorization Machine) combines Factorization Machines (FM) and Deep Neural Networks (DNN) for
recommendation systems. FM captures low-order feature interactions, while DNN models high-order non-linear interactions.
The model is end-to-end trainable and excels in tasks like click-through rate (CTR) prediction and personalized
recommendations. By integrating both FM and DNN, DeepFM efficiently handles sparse data, offering better performance
compared to traditional methods, especially in large-scale applications such as advertising and product recommendations.

## Step 1: Installation

```sh
git clone -b release/2.3.0  https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```sh
pushd datasets/criteo/
sh run.sh
popd
```

## Step 3: Training

```sh
cd models/rank/deepfm
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=3
# train
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
# Eval
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Reference

- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec.git)
