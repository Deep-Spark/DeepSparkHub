# DeepSpeed Llama-2-7B RMF (DeepSpeed)

## Model description

LLaMA2 is a large language model released by Meta in 2023, with parameters ranging from 7B to 70B. Compared to LLaMA,
the training corpus of LLaMA2 is 40% longer, and the context length has been upgraded from 2048 to 4096, allowing for
understanding and generating longer texts.

## Step 1: Installation

```sh
cd deepsparkhub/nlp/llm/llama2-7b_reward_sft/deepspeed
pip install -r requirements.txt
pip install -e .
```

## Step 2: Preparing datasets

```sh
# Install lfs
wget https://packagecloud.io/github/git-lfs/packages/el/7/git-lfs-2.13.2-1.el7.x86_64.rpm/download -O lfs.rpm
rpm -ivh lfs.rpm
rm -y lfs.rpm
git lfs install

# Get Dahoas/rm-static
mkdir -p datasets/Dahoas && cd datasets/Dahoas
git clone https://huggingface.co/datasets/Dahoas/rm-static

# Get Llama-2-7b-hf pretraining weights 
- [Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [aifasthub](https://aifasthub.com/models/NousResearch/Llama-2-7b-hf)
 
mv Llama-2-7b-hf/ datasets/
```

## Step 3: Training

```sh
cd training/step2_reward_model_finetuning/training_scripts/llama2/
bash ./run_llama2_7b.sh
```

## Results

| GPUs       | Epochs | FPS                     | ACC    |
|------------|--------|-------------------------|--------|
| BI-V100 x8 | 1      | AvgSamplesPerSec: 1.948 | 0.6821 |

## Reference

- [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
