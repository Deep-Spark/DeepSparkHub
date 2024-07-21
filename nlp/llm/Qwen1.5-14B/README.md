# Qwen1.5-7B

## Model description
Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:8 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, 32B and 72B dense models, and an MoE model of 14B with 2.7B activated;Significant performance improvement in Chat models;Multilingual support of both base and chat models;Stable support of 32K context length for models of all sizes;No need of trust_remote_code.


## Step 1: Installation

```bash
# install firefly
pushd <deepsparkhub_root>/toolbox/firefly
pip3 install -r requirements.txt
python3 setup.py develop
popd
```

## Step 2: Preparing datasets and checkpoints

```bash
mkdir -p /home/datasets/nlp
git clone -b school_math_0.25M git@gitee.com:sanghui-ilu/datasets.git
mv datasets/school_math_0.25M.jsonl /home/datasets/nlp
rm -rf datasets

mkdir -p /home/model_zoo/nlp
pip install modelscope
python3 ./get_Qwen1.5-7B.py --model=Qwen1.5-14B
mv /root/.cache/modelscope/hub/qwen/Qwen1___5-14B /home/model_zoo/nlp
```

## Step 3: Training
```bash
$ bash train.sh 16 configs/qwen-14b-sft-full.json full  
```

## Results

| No.  | model     | peft        |    num_gpus        |train_samples_per_second |
| ---- | --------- | ----------- | ------------------ | ----------------------  |
| 1    | qwn-14B | Full sft    | 16                 |         2.099          |

## Reference

- [Firefly](https://github.com/yangjianxin1/Firefly)
