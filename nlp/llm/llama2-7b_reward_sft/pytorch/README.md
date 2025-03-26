# Llama-2-7B RMF (DeepSpeed)

## Model Description

Llama-2-7B RMF is a fine-tuned version of Meta's Llama-2-7B model, optimized using DeepSpeed's advanced training
techniques. This model incorporates reward modeling for improved alignment with human preferences, enhancing its
performance in dialogue and instruction-following tasks. With 7 billion parameters and a 4096-token context window, it
excels in understanding and generating coherent, contextually relevant responses. The DeepSpeed optimization enables
efficient training and inference, making it a powerful tool for developing high-quality conversational AI systems while
maintaining computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

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

### Install Dependencies

```sh
cd deepsparkhub/nlp/llm/llama2-7b_reward_sft/pytorch
pip install -r requirements.txt
pip uninstall numpy
pip install numpy==1.23.5
pip install -e .
```

## Model Training

```sh
cd training/step2_reward_model_finetuning/training_scripts/llama2/
bash ./run_llama2_7b.sh
```

## Model Results

| Model          | GPUs       | Epochs | FPS                     | ACC    |
|----------------|------------|--------|-------------------------|--------|
| Llama-2-7B RMF | BI-V100 x8 | 1      | AvgSamplesPerSec: 1.948 | 0.6821 |

## References

- [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
