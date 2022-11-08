# Transformer


## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

```
cd PaddleNLP
pip3 install -r requirements.txt
```

## Step 2: Training
The training is use AMP model.
```
cd PaddleNLP/examples/machine_translation/transformer
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 train.py --config ./configs/transformer.big.yaml \
--use_amp True --amp_level O1
```

## Reference
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)