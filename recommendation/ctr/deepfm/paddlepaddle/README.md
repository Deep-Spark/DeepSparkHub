# DeepFM

## Step 1: Installation

```bash
git clone -b release/2.3.0  https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```bash
pushd datasets/criteo/
sh run.sh
popd
```


## Step 3: Training

```bash
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