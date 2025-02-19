# ChatGLM2-6B SFT (DeepSpeed)

## Model description

This warehouse realizes the fine-tuning of ChatGLM2-6B model based on P-Tuning v2. P-Tuning v2 can reduce the number of
parameters that need to be fine-tuned to 0.1% of the original, and then run with a minimum of 7GB of video memory
through model quantization, Gradient Checkpoint and other methods

## Step 1: Installation

```bash
cd ptuning/
pip3 install -r requirements.txt
```

Downloading a model from Hugging Face Hub requires first [installing Git
LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
and then running

```bash
mkdir -p data
cd data 
git clone https://huggingface.co/THUDM/chatglm2-6b
cd ..
```

## Step 3: Training

### Train

```bash
bash train_ptuning_v2.sh
```

### Test

```bash
bash evaluate_ptuning_v2.sh
```

## Reference

- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)
