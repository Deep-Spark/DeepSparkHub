# Llama3-8B (OpenRLHF)

## Model Description

Llama3-8B is an advanced auto-regressive language model developed by Meta, featuring 8 billion parameters. It utilizes
an optimized transformer architecture with Grouped-Query Attention (GQA) for improved inference efficiency. Trained on
sequences of 8,192 tokens and using a 128K token vocabulary, it excels in various natural language tasks. The model
incorporates supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human
preferences, ensuring both helpfulness and safety in its responses. Llama3-8B offers state-of-the-art performance in
language understanding and generation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.09  |
| BI-V150 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

```sh
git clone https://github.com/OpenRLHF/OpenRLHF.git -b v0.5.7
cd examples/scripts/
mkdir -p OpenRLHF
mkdir -p Dylan2048
# get datasets from huggingface
# for dpo: OpenRLHF/preference_dataset_mixture2_and_safe_pku
# for kto: Dylan2048/ultrafeedback-unpaired-preferences
# for ppo: OpenRLHF/prompt-collection-v0.1
# for sft: Open-Orca/OpenOrca

# get pretrain model from huggingface: https://huggingface.co/OpenRLHF/Llama-3-8b-sft-mixture
# get reward_pretrain model from huggingface: https://huggingface.co/OpenRLHF/Llama-3-8b-rm-mixture
# get model from https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

### Install Dependencies

```sh
# install
cp requirements.txt OpenRLHF/requirements.txt
cd OpenRLHF
pip install -e .
```

## Model Training

```sh
# Make sure you have need 16 BI-V150
cp *.sh OpenRLHF/examples/scripts/
cd OpenRLHF/examples/scripts/
# train with dpo
bash train_dpo_llama.sh

# train with kto
bash train_kto_llama.sh

# train with ppo
bash train_ppo_llama.sh

# train with sft
bash train_sft_llama.sh

# Tips:
# If you throw out: FileNotFoundError: Directory OpenRLHF/prompt-collection-v0.1 is neither a `Dataset` directory nor a `DatasetDict` directory.
# please modify OpenRLHF/openrlhf/utils/utils.py:76 `data = load_from_disk(dataset)` --> `data = load_dataset(dataset, data_dir=data_dir)`
```

## References

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
