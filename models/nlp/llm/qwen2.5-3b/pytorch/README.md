# Qwen2.5-3B (ColossalAI)

## Model Description

Qwen2.5 is an advanced large language model series developed by Alibaba Cloud, offering significant improvements over
its predecessor. With enhanced capabilities in coding, mathematics, and structured data processing, it supports context
lengths up to 128K tokens and generates outputs up to 8K tokens. The model excels in multilingual support across 29
languages and demonstrates robust performance in instruction following and role-play scenarios. Qwen2.5's optimized
architecture and specialized expert models make it a versatile tool for diverse AI applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

```sh
git clone -b v0.4.8 https://github.com/hpcaitech/ColossalAI.git --depth=1
cd ColossalAI/
cp -rf <DeepSparkHub_Root>/toolbox/ColossalAI/v0.4.8/patches/* ./
cd applications/ColossalChat/examples/
# get qwen2.5-3b from https://huggingface.co/Qwen/Qwen2.5-3B and put it in checkpoints/Qwen2.5-3B
mkdir -p checkpoints
# get qwedsacf/competition_math dataset and put it in datasets/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet
mkdir -p dataset/competition_math/data
mkdir -p dataset/competition_math/sft
```

### Install Dependencies

```sh
pip install wheel
pip install transformers==4.39.3
pip install http://files.deepspark.org.cn:880/deepspark/add-ons/bitsandbytes-0.43.3+corex.4.3.0-cp310-cp310-linux_x86_64.whl

cd ColossalAI/
bash clean_colossalai.sh
bash build_colossalai.sh
bash install_colossalai.sh

cd applications/ColossalChat
pip install -e .

cd examples/data_preparation_scripts
mkdir -p processed_data/sft
python3 process_competition_math.py
bash prepare_sft_dataset.sh
```

## Model Training

```sh
# pls remove the redundant /usr/local/corex-4.3.0/include/torch folder before you run.
cd examples/training_scripts
bash train_sft_Qwen.sh
```

## Model Results

| Model      | GPUs        | type | train_samples_per_second |
|------------|-------------|------|--------------------------|
| Qwen2.5-3b | BI-V150 x 4 | full | 1.889                    |

## References

- [ColossalAI](https://github.com/hpcaitech/ColossalAI)
