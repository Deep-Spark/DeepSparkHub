# Mixtral 8x7B (OpenRLHF)

## Model Description

The Mixtral model is a Mixture of Experts (MoE)-based large language model developed by Mistral AI, an innovative
company focusing on open-source AI models. Mixtral is designed to achieve high performance while maintaining
computational efficiency, making it an excellent choice for real-world applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

```sh
git clone https://github.com/OpenRLHF/OpenRLHF.git -b v0.5.7
cd examples/scripts/
# get datasets from huggingface Open-Orca/OpenOrca
# get model from https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

```

### Install Dependencies

```sh
# install
cp requirements.txt OpenRLHF/requirements.txt
cd OpenRLHF
pip install -e .
```

## Model Training

```sh
# Make sure you have need 16 BI-V150
cp *.sh OpenRLHF/examples/scripts/
cd OpenRLHF/examples/scripts/
bash train_sft_mixtral_lora.sh
```

## References

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
