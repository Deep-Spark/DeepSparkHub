#  xDeepFM

## Model description
xDeepFM proposes a novel network named Compressed Interaction Network (CIN), which aims to learn high-order feature interactions explicitly. xDeepFM can automatically learn high-order feature interactions in both explicit and implicit fashions, which is of great significance to reducing manual feature engineering work.

## Step 1: Installation

```bash
git clone -b release/2.3.0 https://github.com/PaddlePaddle/PaddleRec.git
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
cd models/rank/xdeepfm

# Training
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml

# Evaluation
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Results
| GPUs        | IPS       | AUC         |
|-------------|-----------|-------------|
| BI-V100 x1  | 6000      | 0.7955      |

## Reference

- [xDeepFM](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/xdeepfm)
