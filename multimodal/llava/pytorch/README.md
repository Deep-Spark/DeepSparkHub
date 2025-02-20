# LLaVA 1.5

## Model description

LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal
instruction-following data. It is an auto-regressive language model, based on the transformer
architecture.

## Step 1: Preparation

```bash
# Clone LLaVA Repo
git clone --depth 1 https://github.com/haotian-liu/LLaVA
cd LLaVA/
git checkout c121f0432da27facab705978f83c4ada465e46fd

# Prepare dirs.
mkdir -p checkpoints/
mkdir -p data/
```

### Weights

Download the [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) and put it
at `checkpoints/`.

Download the [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
and put it at `checkpoints/`.

### Datasets

Download the [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
and put it at `data/`.

## Step 2: Installation

```bash
pip3 install -e .
pip3 install -e ".[train]"
pip3 install protobuf
```

## Step 3: Training

```bash
mv ../train.sh ./
bash train.sh
```

## Results

| Model        | GPUs    | seconds per iteration |
| ------------ | ------- | --------------------- |
| LLaVA 1.5 7B | BI-V150 | 8.01s                 |

## Reference

- [LLaVA](https://github.com/haotian-liu/LLaVA)
