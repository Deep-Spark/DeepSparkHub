# LLaVA 1.5

## Model Description

LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal
instruction-following data. It is an auto-regressive language model, based on the transformer
architecture.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Install Dependencies

```bash
# Clone LLaVA Repo
git clone --depth 1 https://github.com/haotian-liu/LLaVA
cd LLaVA/
git checkout c121f0432da27facab705978f83c4ada465e46fd
pip3 install -e .
pip3 install -e ".[train]"
pip3 install protobuf
```

### Prepare Resources

```bash
# Prepare dirs.
mkdir -p checkpoints/
mkdir -p data/
```

#### Weights

Download the [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) and put it
at `checkpoints/`.

Download the [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
and put it at `checkpoints/`.

#### Datasets

Download the [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
and put it at `data/`.


## Model Training

```bash
mv ../train.sh ./
bash train.sh
```

## Model Results

| Model        | GPUs    | seconds per iteration |
| ------------ | ------- | --------------------- |
| LLaVA 1.5 7B | BI-V150 | 8.01s                 |

## References

- [LLaVA](https://github.com/haotian-liu/LLaVA)
