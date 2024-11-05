# Qwen2.5-7B

## Model description

Qwen2.5 is the latest series of Qwen large language models. Qwen2.5 brings the
following improvements upon Qwen2:

- Significantly more knowledge and has greatly improved capabilities in coding
and mathematics, thanks to our specialized expert models in these domains.
- Significant improvements in instruction following, generating long texts (over
8K tokens), understanding structured data (e.g, tables), and generating
structured outputs especially JSON. More resilient to the diversity of system
prompts, enhancing role-play implementation and condition-setting for chatbots.
- Long-context Support up to 128K tokens and can generate up to 8K tokens.
- Multilingual support for over 29 languages, including Chinese, English,
French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean,
Vietnamese, Thai, Arabic, and more.

## Step 1: Installation

```bash
git clone -b 6b62550af1acda93246e05b37061f2ad7db58e55 --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cp qwen2_5-7b_full_sft.yaml LLaMA-Factory/examples/train_full/
cp qwen2_5-7b_lora_sft.yaml LLaMA-Factory/examples/train_lora/
cd LLaMA-Factory/
pip3 install -r requirements.txt
pip3 install --no-deps -e .
```

## Step 2: Preparing datasets

```bash
# get qwen2.7-7b from https://huggingface.co/Qwen/Qwen2.5-7B and put it
# in checkpoints/Qwen2.5-7B
mkdir -p checkpoints
```

## Step 3: Training

### Full SFT

```bash
llamafactory-cli train examples/train_full/qwen2_5-7b_full_sft.yaml
```

### LoRA SFT

```bash
llamafactory-cli train examples/train_lora/qwen2_5-7b_lora_sft.yaml
```

## Results

|  GPUs   |             Model              | type |train_samples_per_second |
| :-----: | :----------------------------: | :------------: | :------------: |
| BI-V150 x 8 | Qwen2.5-7b|  full           |      1.889  |

## Reference

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
