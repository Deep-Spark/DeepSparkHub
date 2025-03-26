# Mixtral 8x7B (Megatron-LM)

## Model Description

The Mixtral model is a Mixture of Experts (MoE)-based large language model developed by Mistral AI, an innovative
company focusing on open-source AI models. Mixtral is designed to achieve high performance while maintaining
computational efficiency, making it an excellent choice for real-world applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources

```sh
# prepare Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
(cd Megatron-LM && git checkout bd677bfb13ac2f19deaa927adc6da6f9201d66aa)
cp -r -T ../../../../toolbox/Megatron-LM/patch ./Megatron-LM/

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
cd examples/mixtral
bash train_mixtral_8x7b_distributed.sh
```

## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mixtral)
