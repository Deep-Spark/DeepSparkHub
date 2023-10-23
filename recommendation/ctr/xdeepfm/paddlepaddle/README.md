#  xDeepFM
## Model description
xDeepFM proposes a novel network named Compressed Interaction Network (CIN), which aims to learn high-order feature interactions explicitly. xDeepFM can automatically learn high-order feature interactions in both explicit and implicit fashions, which is of great significance to reducing manual feature engineering work.

## Step 1: Installation

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```bash
cd PaddleRec
cd datasets/criteo/
sh run.sh
```

## Step 3: Training

```bash
cd models/rank/xdeepfm

# Training
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml

# Evaluation
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Result
| GPUs        | IPS       | AUC         |
|-------------|-----------|-------------|
| BI-V100 x1  | 6000      | 0.7955      |

## Reference
- [xDeepFM](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/xdeepfm)
