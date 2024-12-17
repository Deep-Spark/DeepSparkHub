# Mamba-2 （Megatron-LM)

## Model description

Mamba-2 is a cutting-edge state space model (SSM) architecture designed as a highly efficient alternative to traditional Transformer-based large language models (LLMs). It is the second version of the Mamba model and builds on the strengths of its predecessor by offering faster inference, improved scalability for long sequences, and lower computational complexity.

## Step 1: Installation

```sh
# uninstall
pip3 uninstall -y megatron-lm

# clone and install
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/ && git checkout bd677bfb13ac2f19deaa927adc6da6f9201d66aa
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
cd examples/mamba
bash train.sh
```

## Reference

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mamba)
