# FFM 

## Model description
FFM is further improved on the basis of FM, introducing the concept of category in the model, that is, field. 
The features of the same field are one-hot separately, so in FFM, each one-dimensional feature learns a hidden variable for each field of the other features, which is not only related to the feature, but also to the field. 
By introducing the concept of field, FFM attributes features of the same nature to the same field.

## Step 1: Installing
```
git clone -b release/2.3.0 https://github.com/PaddlePaddle/PaddleRec.git
```

```
cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Training
```

# Download dataset
cd datasets/criteo/
sh run.sh


cd  ../../models/rank/ffm
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0
# train
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
# Eval
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Result
|  GPU       |  AUC      |  IPS |   
|---         |---        |---          |
| BI-V100 x1 |   0.792128| 714.42ins/s    |   



## Reference
- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)
