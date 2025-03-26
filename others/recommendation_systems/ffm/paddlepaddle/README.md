# FFM

## Model Description

FFM is further improved on the basis of FM, introducing the concept of category in the model, that is, field. The
features of the same field are one-hot separately, so in FFM, each one-dimensional feature learns a hidden variable for
each field of the other features, which is not only related to the feature, but also to the field. By introducing the
concept of field, FFM attributes features of the same nature to the same field.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

```bash
# Prepare PaddleRec
git clone -b release/2.3.0 https://github.com/PaddlePaddle/PaddleRec.git

# Download dataset
pushd PaddleRec/datasets/criteo/
sh run.sh
popd
```

### Install Dependencies

```sh
cd PaddleRec/
pip3 install -r requirements.txt
```

## Model Training

```sh
cd  models/rank/ffm
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0
# train
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
# Eval
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Model Results

| Model | GPU        | AUC      | IPS         |
|-------|------------|----------|-------------|
| FFM   | BI-V100 x1 | 0.792128 | 714.42ins/s |

## References

- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)
