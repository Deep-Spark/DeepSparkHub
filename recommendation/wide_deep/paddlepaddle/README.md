# Wide&Deep


## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleRec.git
```

```
cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Training
```
cd PaddleRec

# 下载数据集
pushd datasets/criteo/
sh run.sh
popd

pushd models/rank/wide_deep
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=3
# train
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
# Eval
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
popd
```

## Reference
- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)