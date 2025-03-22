# Yi-VL-6B SFT (LLaMA-Factory)

## Model description
    Yi Visual Language (Yi-VL) model is the open-source, multimodal version of the Yi Large Language Model (LLM) series, enabling content comprehension, recognition, and multi-round conversations about images. 
    Yi-VL demonstrates exceptional performance, ranking first among all existing open-source models in the latest benchmarks including MMMU in English and CMMMU in Chinese (based on data available up to January 2024).

    Yi-VL offers the following features:
        Multi-round text-image conversations: Yi-VL can take both text and images as inputs and produce text outputs. Currently, it supports multi-round visual question answering with one image.
        Bilingual text support: Yi-VL supports conversations in both English and Chinese, including text recognition in images.
        Strong image comprehension: Yi-VL is adept at analyzing visuals, making it an efficient tool for tasks like extracting, organizing, and summarizing information from images.
        Fine-grained image resolution: Yi-VL supports image understanding at a higher resolution of 448×448.

## Step 1: Installation

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory/
git checkout d8a5571be7fcdc6f9e2442a832252d507f58c862
cp ../yi_vl_6b_full_sft.yaml examples/train_full/
cp ../yi_vl_6b_lora_sft.yaml examples/train_lora/
pip3 install -r requirements.txt
```

## Step 2: Preparing model

```sh
mkdir -p /home/model_zoos/nlp/Yi-VL-6B-hf

# download model Yi-VL-6B-hf (https://huggingface.co/BUAADreamer/Yi-VL-6B-hf), and then put it in /home/model_zoos/nlp/Yi-VL-6B-hf

```

## Step 3: Training

```sh
 export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
```

### Full SFT
```sh
llamafactory-cli train examples/train_full/yi_vl_6b_full_sft.yaml
```

### Lora SFT
```sh
llamafactory-cli train examples/train_lora/yi_vl_6b_lora_sft.yaml
```

## Results

| GPUs        | Model      | type | train_samples_per_second |
|-------------|------------|------|--------------------------|
| BI-V150 x 8 | Yi-VL-6B   | full | 0.546                    |
| BI-V150 x 8 | Yi-VL-6B   | lora | 2.474                    |


## Reference

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
