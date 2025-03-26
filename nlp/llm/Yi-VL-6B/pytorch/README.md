# Yi-VL-6B SFT (LLaMA-Factory)

## Model Description

Yi-VL-6B is an advanced multimodal language model developed by 01.AI, designed for visual-language understanding and
interaction. With 6 billion parameters, it excels in tasks like image comprehension, text recognition, and multi-round
conversations involving both text and images. Supporting English and Chinese, Yi-VL-6B demonstrates state-of-the-art
performance in benchmarks like MMMU and CMMMU. Its ability to process high-resolution images (448×448) and engage in
detailed visual question answering makes it a powerful tool for AI-driven image-text analysis and dialogue systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

```sh
mkdir -p /home/model_zoos/nlp/Yi-VL-6B-hf
```

Download model [Yi-VL-6B-hf](https://huggingface.co/BUAADreamer/Yi-VL-6B-hf), and then put it in
/home/model_zoos/nlp/Yi-VL-6B-hf.

### Install Dependencies

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory/
git checkout d8a5571be7fcdc6f9e2442a832252d507f58c862

cp ../yi_vl_6b_full_sft.yaml examples/train_full/
cp ../yi_vl_6b_lora_sft.yaml examples/train_lora/
pip3 install -r requirements.txt
```

## Model Training

```sh
 export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1

# Full SFT
llamafactory-cli train examples/train_full/yi_vl_6b_full_sft.yaml

# Lora SFT
llamafactory-cli train examples/train_lora/yi_vl_6b_lora_sft.yaml
```

## Model Results

| Model    | GPU        | type | train_samples_per_second |
|----------|------------|------|--------------------------|
| Yi-VL-6B | BI-V150 x8 | full | 0.546                    |
| Yi-VL-6B | BI-V150 x8 | lora | 2.474                    |

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
