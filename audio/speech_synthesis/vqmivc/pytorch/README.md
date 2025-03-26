# VQMIVC

## Model Description

VQMIVC is an advanced deep learning model for one-shot voice conversion that employs vector quantization and mutual
information minimization to disentangle speech representations. It effectively separates content, speaker, and pitch
information, enabling high-quality voice conversion with just a single target-speaker utterance. By reducing
inter-dependencies between speech components, VQMIVC achieves superior naturalness and speaker similarity compared to
traditional methods. This unsupervised approach is particularly effective for retaining source linguistic content while
accurately capturing target speaker characteristics.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```sh
mkdir -p /home/data/vqmivc/
cd /home/data/vqmivc/
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip
```

### Install Dependencies

```sh
pip3 install -r requirements_bi.txt
ln -s /home/data/vqmivc .
python3 preprocess.py
ln -s vqmivc/data .
```

## Model Training

- Training with mutual information minimization (MIM):

```sh
export HYDRA_FULL_ERROR=1
python3 train.py use_CSMI=True use_CPMI=True use_PSMI=True
```

- Training without MIM:

```sh
python3 train.py use_CSMI=False use_CPMI=False use_PSMI=False 
```

## Model Results

| Card Type | recon loss | cps loss | vq loss | perpexlity | lld cs loss | mi cs loss | lld ps loss | mi ps loss | lld cp loss | mi cp loss | used time(s) |
|-----------|------------|----------|---------|------------|-------------|------------|-------------|------------|-------------|------------|--------------|
| BI-V100   | 0.635      | 1.062    | 0.453   | 401.693    | 110.958     | 2.653E-4   | 0.052       | 0.001      | 219.895     | 0.021      | 4.315        |
|           |

## References

- [VQMIVC](https://github.com/Wendison/VQMIVC)
