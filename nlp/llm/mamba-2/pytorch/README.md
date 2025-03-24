# Mamba-2 （Megatron-LM)

## Model Description

Mamba-2 is a cutting-edge state space model (SSM) architecture designed as a highly efficient alternative to traditional
Transformer-based large language models (LLMs). It is the second version of the Mamba model and builds on the strengths
of its predecessor by offering faster inference, improved scalability for long sequences, and lower computational
complexity.

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
