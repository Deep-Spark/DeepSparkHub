# Facenet

## Model Description

This is a facenet-pytorch library that can be used to train your own face recognition model.
> Despite significant recent advances in the field of face recognition, implementing face
verification and recognition efficiently at scale presents serious challenges to current
approaches. In this paper we present a system, called FaceNet, that directly learns a mapping
from face images to a compact Euclidean space where distances directly correspond to a measure
of face similarity. Once this space has been produced, tasks such as face recognition,
verification and clustering can be easily implemented using standard techniques with FaceNet
embeddings as feature vectors.

## Step 1: Installation

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

## Step 2: Preparing datasets

You can download datasets from [BaiduPan](https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw) with password 'bcrq'.

```bash
# download
ln -s ${your_path_to_face} datasets/
ln -s ${your_path_to_face} lfw/
ln -s ${your_path_to_face} lfw_pair.txt

# preprocess
python3 txt_annotation.py
```

## Step 3: Training

```bash
python3 train.py
```

## Results

|   model |    FPS | LFW_Accuracy     |
|---------|--------| -----------------|
| facenet | 1256.96| 0.97933+-0.00624 |

## Reference
- [paper](https://arxiv.org/abs/1503.03832)
- [facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch)
