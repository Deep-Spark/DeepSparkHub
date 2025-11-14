# Qwen2.5-VL-7B (LLaMA-Factory)

## Model Description

Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images.
Directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use. Can comprehend videos of over 1 hour, and this time it has a new ability of cpaturing event by pinpointing the relevant video segments. Can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes. For data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0  |  25.12  |

## Model Preparation

### Prepare Resources

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory/
git checkout 8173a88a26a1cfe78738a826047d1ef923cd4ea3
mkdir -p Qwen
# download https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct into Qwen folder
mkdir -p llamafactory
# dowload https://huggingface.co/datasets/llamafactory/RLHF-V into llamafactory folder
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- accelerate-0.34.2+corex.4.3.0-py3-none-any.whl

```sh
pip install llamafactory==0.9.2
pip install peft==0.11.1
pip install transformers==4.49.0
```

## Model Training

```bash
# please set val_size with 0.01 in yaml to disable eval
# dpo
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_dpo.yaml
# sft
pip install 
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml
```

## Model Results

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git)

