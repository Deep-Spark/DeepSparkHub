# Qwen3 (LoongForge)

## Model Description

LoongForge is a unified training framework for LLMs, VLMs, VLAs, and diffusion models, covering pre-training, continued pre-training, and SFT. Built upon Megatron-LM with deep systemic enhancements across model coverage, training performance, and hardware support, it delivers significant speedups over mainstream open-source baselines.

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support, with the following key features:

- Uniquely support of seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue) within single model, ensuring optimal performance across various scenarios.
- Significantly enhancement in its reasoning capabilities, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
- Superior human preference alignment, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
- Expertise in agent capabilities, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
- Support of 100+ languages and dialects with strong capabilities for multilingual instruction following and translation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.4.0  |  26.06 |

## Model Preparation

### Prepare Resources

```sh
mkdir -p /workspace
cd /workspace
git clone -b v0.1.0 --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge
pip install -e ".[gpu]"
ln -s /workspace/LoongForge/third_party/Loong-Megatron /workspace/Loong-Megatron
cd examples/qwen3
mkdir -p dataset
hf download yahma/alpaca-cleaned --repo-type dataset --local-dir ./dataset/alpaca-cleaned
hf download wikitext --repo-type dataset --include "wikitext-103-raw-v1*" --local-dir ./dataset/wikitext
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/covert_wiktext_dataset.py ./
python covert_wiktext_dataset.py

mkdir -p checkpoints
modelscope download --model Qwen/Qwen3-0.6B --local_dir checkpoints/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-4B --local_dir checkpoints/Qwen3-4B
modelscope download --model Qwen/Qwen3-8B --local_dir checkpoints/Qwen3-8B
```

### Install Dependencies

```bash
ln -s /usr/local/bin/python3.10 /usr/local/bin/python
pip uninstall megatron-deepspeed ixmegatron megatron-lm flashinfer-python

# Delete histoty key-value
sed -i '7 s/history: history//' /workspace/LoongForge/configs/data/sft_dataset_config.yaml
sed -i '15 s/history: history//' /workspace/LoongForge/configs/data/sft_dataset_config.yaml

cd examples/qwen3/finetuning
## Prepare SFT Datasets
export TOKENIZER_PATH=/workspace/LoongForge/examples/qwen3/checkpoints/Qwen3-0.6B
# Change input_data and output_path path in preprocess_data.sh
sed -i '8 s/\/mnt\/cluster\/LoongForge\/dataset\/sft_aplaca_zh_data.json/..\/dataset\/alpaca-cleaned\/alpaca_data_cleaned.json/' preprocess_data.sh
sed -i '9 s/\/mnt\/cluster\/LoongForge\/qwen3\/sft_aplaca_zh_tokenized/..\/dataset\/alpaca-cleaned\/save_dir/' preprocess_data.sh
bash preprocess_data.sh

cd examples/qwen3/pretrain
## Prepare Pretrain Datasets
# Change input_data and output_path path in preprocess_data.sh
sed -i '8 s/\/mnt\/cluster\/LoongForge\/dataset\/pile_test\/train.jsonl/..\/dataset\/wikitext_train.jsonl/' preprocess_data.sh
sed -i '9 s/\/mnt\/cluster\/LoongForge\/qwen3\/pile_test\/pile-qwen/..\/dataset\/pile-qwen/' preprocess_data.sh
bash preprocess_data.sh

## Adapt checkpoint convert
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/checkpointing.py /workspace/LoongForge/loongforge/train/checkpointing.py
```

## Model Training
### Qwen3-0.6B
```bash
## Prepare Checkpoints
cd ../checkpoint_convert/
# Change LOAD and SAVE path in convert_qwen3_0.6b_hf_to_mcore.sh
sed -i '7 s/\/mnt\/cluster\/models\/Qwen3-0.6B/..\/checkpoints\/Qwen3-0.6B/' convert_qwen3_0.6b_hf_to_mcore.sh
sed -i '8 s/\/mnt\/cluster\/LoongForge\/qwen3\/qwen3-0.6b-tp1-pp1-Dec24/..\/checkpoints\/qwen3-0.6b-mcore/' convert_qwen3_0.6b_hf_to_mcore.sh
bash convert_qwen3_0.6b_hf_to_mcore.sh
cd ../finetuning
export TOKENIZER_PATH=/workspace/LoongForge/examples/qwen3/checkpoints/Qwen3-0.6B
export DATA_PATH=../dataset/alpaca-cleaned/save_dir
export DATASET_CONFIG_PATH=/workspace/LoongForge/configs/data/sft_dataset_config.yaml
export CHECKPOINT_PATH=../checkpoints/qwen3-0.6b-mcore
export TENSORBOARD_PATH=./tensorboard-log/qwen3-0.6b-sft
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/sft_qwen3_0.6b.sh ./
bash sft_qwen3_0.6b.sh

cd ../pretrain
export DATA_PATH=../dataset/pile-qwen_text_document
export CHECKPOINT_PATH=../checkpoints/qwen3-0.6b-mcore
export TENSORBOARD_PATH=./tensorboard-log/qwen3-0.6b-pretrain
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/pretrain_qwen3_0.6b.sh ./
bash pretrain_qwen3_0.6b.sh
```

### Qwen3-4B
```bash
## Prepare Checkpoints
cd ../checkpoint_convert/
# Change LOAD and SAVE path in convert_qwen3_4b_hf_to_mcore.sh
sed -i '7 s/\/mnt\/cluster\/huggingface.co\/Qwen\/Qwen3-4B/..\/checkpoints\/Qwen3-4B/' convert_qwen3_4b_hf_to_mcore.sh
sed -i '8 s/\/mnt\/cluster\/LoongForge\/qwen3\/qwen3-4b-tp1-pp1-Dec24/..\/checkpoints\/qwen3-4b-mcore/' convert_qwen3_4b_hf_to_mcore.sh
sed -i '14 s/=1/=2/' convert_qwen3_4b_hf_to_mcore.sh
sed -i '15 s/=1/=2/' convert_qwen3_4b_hf_to_mcore.sh
bash convert_qwen3_4b_hf_to_mcore.sh
cd ../finetuning
export DATA_PATH=../dataset/alpaca-cleaned/save_dir
export DATASET_CONFIG_PATH=/workspace/LoongForge/configs/data/sft_dataset_config.yaml
export CHECKPOINT_PATH=../checkpoints/qwen3-4b-mcore
export TENSORBOARD_PATH=./tensorboard-log/qwen3-4b-sft
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/sft_qwen3_4b.sh ./
bash sft_qwen3_4b.sh

cd ../pretrain
export DATA_PATH=../dataset/pile-qwen_text_document
export CHECKPOINT_PATH=../checkpoints/qwen3-4b-mcore
export TENSORBOARD_PATH=./tensorboard-log/qwen3-4b-pretrain
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/pretrain_qwen3_4b.sh ./
bash pretrain_qwen3_4b.sh
```

### Qwen3-8B
```bash
## Prepare Checkpoints
cd ../checkpoint_convert/
# Change LOAD and SAVE path in convert_qwen3_8b_hf_to_mcore.sh
sed -i '7 s/\/mnt\/cluster\/huggingface.co\/Qwen\/Qwen3-8B/..\/checkpoints\/Qwen3-8B/' convert_qwen3_8b_hf_to_mcore.sh
sed -i '8 s/\/mnt\/cluster\/LoongForge\/qwen3\/qwen3-8b-tp1-pp1-Dec24/..\/checkpoints\/qwen3-8b-mcore/' convert_qwen3_8b_hf_to_mcore.sh
sed -i '14 s/=1/=4/' convert_qwen3_8b_hf_to_mcore.sh
bash convert_qwen3_8b_hf_to_mcore.sh
cd ../finetuning
export DATA_PATH=../dataset/alpaca-cleaned/save_dir
export DATASET_CONFIG_PATH=/workspace/LoongForge/configs/data/sft_dataset_config.yaml
export CHECKPOINT_PATH=../checkpoints/qwen3-8b-mcore
export TENSORBOARD_PATH=./tensorboard-log/qwen3-8b-sft
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/sft_qwen3_8b.sh ./
bash sft_qwen3_8b.sh

cd ../pretrain
export DATA_PATH=../dataset/pile-qwen_text_document
export CHECKPOINT_PATH=../checkpoints/qwen3-8b-mcore
export TENSORBOARD_PATH=./tensorboard-log/qwen3-8b-pretrain
cp /Path/to/DeepSparkInference/models/nlp/llm/qwen3/loongforge/pretrain_qwen3_8b.sh ./
bash pretrain_qwen3_8.sh
```

## References

- [LoongForge](https://github.com/baidu-baige/LoongForge)
