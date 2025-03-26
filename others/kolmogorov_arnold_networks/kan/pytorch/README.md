# KAN

## Model Description

Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong
mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on
Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs
have activation functions on nodes. This simple change makes KANs better (sometimes much better!) than MLPs in terms of
both model accuracy and interpretability.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

```bash
bash ./run_train.sh
```

## Model Results

| Model | Training speed   |
|-------|------------------|
| KAN   | 6490 samples/sec |

## References

- [pykan](https://github.com/KindXiaoming/pykan/tree/master)
