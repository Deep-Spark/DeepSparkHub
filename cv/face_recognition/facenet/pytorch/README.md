# Facenet

## Model Description

Facenet is a deep learning model for face recognition that directly maps face images to a compact Euclidean space, where
distances correspond to face similarity. It uses a triplet loss function to ensure that faces of the same person are
closer together than those of different individuals. Facenet excels in tasks like face verification, recognition, and
clustering, offering high accuracy and efficiency. Its compact embeddings make it scalable for large-scale applications
in security and identity verification.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

You can download datasets from [BaiduPan](https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw) with password 'bcrq'.

```bash
# download
ln -s ${your_path_to_face} datasets/
ln -s ${your_path_to_face} lfw/
ln -s ${your_path_to_face} lfw_pair.txt

# preprocess
python3 txt_annotation.py
```

### Install Dependencies

```bash
pip3 install -r requirements.txt

# install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install

# install other requirements
pip3 install matplotlib
pip3 install scikit-learn
```

### Model Training

```bash
python3 train.py
```

## Model Results

| Model   | FPS     | LFW_Accuracy     |
|---------|---------|------------------|
| Facenet | 1256.96 | 0.97933+-0.00624 |

## References

- [Paper](https://arxiv.org/abs/1503.03832)
- [facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch)
