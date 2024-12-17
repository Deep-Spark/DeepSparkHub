# Mixtral 8x7B (Megatron-LM)

## Model description

The Mixtral model is a Mixture of Experts (MoE)-based large language model developed by Mistral AI, an innovative company focusing on open-source AI models. Mixtral is designed to achieve high performance while maintaining computational efficiency, making it an excellent choice for real-world applications.

## Step 1: Installation

```sh
# uninstall
pip3 uninstall -y megatron-lm

# clone and install
git clone https://github.com/NVIDIA/Megatron-LM.git
(cd Megatron-LM/ && git checkout bd677bfb13ac2f19deaa927adc6da6f9201d66aa)
## apply patch
cp -r -T ../../../../toolbox/Megatron-LM/patch ./Megatron-LM/
## install
cd Megatron-LM/
python3 setup.py develop
```

## Step 2: Preparing datasets

```sh
cd datasets/
bash download_and_convert_dataset.sh
```

## Step 3: Training

```bash
cd examples/mixtral
bash train_mixtral_8x7b_distributed.sh
```

## Reference

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mixtral)
