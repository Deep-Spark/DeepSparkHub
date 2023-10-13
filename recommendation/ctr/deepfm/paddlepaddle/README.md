# DeepFM

## Step 1: Installing

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Download data

```bash
cd PaddleRec
cd datasets/criteo/
sh run.sh
```


## Step 3: Run DeepFM

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