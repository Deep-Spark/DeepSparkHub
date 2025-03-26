# ChatGLM2-6B SFT (DeepSpeed)

## Model Description

ChatGLM2-6B SFT is an enhanced version of the ChatGLM2-6B model, fine-tuned using P-Tuning v2 for efficient adaptation
to specific tasks. This approach reduces the number of trainable parameters to just 0.1% of the original model, enabling
fine-tuning with minimal computational resources. Through techniques like model quantization and gradient checkpointing,
it can operate on GPUs with as little as 7GB of memory. ChatGLM2-6B SFT maintains the original model's bilingual
capabilities while offering improved task-specific performance and resource efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 3.4.0     |  24.06  |

## Model Preparation

### Prepare Resources

Downloading a model from Hugging Face Hub requires first [installing Git
LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
and then running

```bash
cd ptuning/
mkdir -p data
cd data 
git clone https://huggingface.co/THUDM/chatglm2-6b
cd ..
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

```bash
# Train
bash train_ptuning_v2.sh

# Test
bash evaluate_ptuning_v2.sh
```

## References

- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)
