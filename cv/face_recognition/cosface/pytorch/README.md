# CosFace

## Model Description

CosFace is a face recognition model that achieves state-of-the-art results by introducing a cosine margin penalty in the
loss function when training the neural network, which learns highly discriminative facial embeddings by maximizing
inter-class differences and minimizing intra-class variations.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

You can download datasets from [BaiduPan](https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw) with password 'bcrq'.

```bash
# download
mkdir datasets
ln -s ${your_path_to_face} datasets ./datasets
ln -s ${your_path_to_face} lfw .
ln -s ${your_path_to_face} lfw_pair.txt .

# preprocess
python3 txt_annotation.py
```

### Install Dependencies

```bash
# install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
```

## Model Training

```bash
# 1 GPU
bash train.sh 0

# 8 GPUs
bash train.sh 0,1,2,3,4,5,6,7
```

## Model Results

| Model   | FPS     | LFW_Accuracy |
|---------|---------|--------------|
| Cosface | 5492.08 | 0.9865       |

## References

- [CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch)
