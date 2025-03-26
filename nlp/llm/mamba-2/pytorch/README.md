# Mamba-2 （Megatron-LM)

## Model Description

Mamba-2 is a cutting-edge state space model (SSM) architecture designed as a highly efficient alternative to traditional
Transformer-based large language models (LLMs). It is the second version of the Mamba model and builds on the strengths
of its predecessor by offering faster inference, improved scalability for long sequences, and lower computational
complexity.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources

```sh
# prepare Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
(cd Megatron-LM && git checkout bd677bfb13ac2f19deaa927adc6da6f9201d66aa)
cp -r -T ../../../../toolbox/Megatron-LM/patch ./Megatron-LM/

# download dataset
pushd Megatron-LM/datasets/
bash download_and_convert_dataset.sh
popd
```

### Install Dependencies

```sh
# uninstall
pip3 uninstall -y megatron-lm

## install
cd Megatron-LM/
python3 setup.py develop
```

## Model Training

```sh
cd examples/mamba
bash train.sh
```

## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mamba)
