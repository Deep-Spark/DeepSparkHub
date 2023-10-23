#  xDeepFM
## Model description
xDeepFM propose a novel network named Compressed Interaction Network (CIN), which aims to learn high-order feature interactions explicitly. xDeepFM can automatically learn high-order feature interactions in both explicit and implicit fashions, which is of great significance to reducing manual feature engineering work.
## Step 1: Installing

```
git clone --recursive https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```
cd PaddleRec
cd datasets/criteo/
sh run.sh
```

## Step 3: Training

```
cd models/rank/xdeepfm
# train
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
# Eval
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Result
| GPUs        | IPS       | AUC         |
|-------------|-----------|-------------|
| BI-V100x1   | 6000      | 0.7955      |
