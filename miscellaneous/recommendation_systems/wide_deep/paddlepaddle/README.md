# Wide&Deep

## Model description

"Wide & Deep Learning" is a machine learning model architecture that combines the strengths of both memorization and generalization. It was introduced by Google in the context of recommender systems, particularly for improving the performance of large-scale recommendation tasks.

## Step 1: Installation

```bash
git clone -b release/2.3.0 https://github.com/PaddlePaddle/PaddleRec.git

cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```bash
# Download dataset
pushd datasets/criteo/
sh run.sh
popd
```

## Step 3: Training

```bash
cd models/rank/wide_deep
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0

# Training
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml

# Evaluation
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Reference
- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)
