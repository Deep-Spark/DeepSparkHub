# CosFace

## Model description

CosFace is a face recognition model that achieves state-of-the-art results by introducing a cosine margin penalty in the loss function when training the neural network, which learns highly discriminative facial embeddings by maximizing inter-class differences and minimizing intra-class variations.

## Step 1: Installation

```bash
# install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
```

## Step 2: Preparing datasets

You can download datasets from [BaiduPan](https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw) with password 'bcrq'.

```bash
# download
ln -s ${your_path_to_face} datasets ./datasets
ln -s ${your_path_to_face} lfw .
ln -s ${your_path_to_face} lfw_pair.txt .

# preprocess
python3 txt_annotation.py
```

## Step 3: Training

```bash
# 1 GPU
bash train.sh 0

# 8 GPUs
bash train.sh 0,1,2,3,4,5,6,7
```

## Results

|   model |    FPS | LFW_Accuracy     |
|---------|--------| -----------------|
| Cosface | 5492.08 | 0.9865          |

## Reference
- [CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch)


