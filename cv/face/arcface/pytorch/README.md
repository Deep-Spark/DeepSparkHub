# ArcFace

## Model description
This repo is a pytorch implement of ArcFace, which propose an Additive Angular Margin Loss to obtain highly discriminative features for face recognition. The proposed ArcFace has a clear geometric interpretation due to the exact correspondence to the geodesic distance on the hypersphere. ArcFace consistently outperforms the state-of-the-art and can be easily implemented with negligible computational overhead

## Step 1: Installation

```bash
pip3 install -r requirements.txt
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
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
bash run.sh $GPUS
```

## Results

|   model |    FPS | LFW_Accuracy     |
|---------|--------| -----------------|
| arcface | 38.272 | 0.99000+-0.00615 |

## Reference
- [arcface-pytorch](https://github.com/bubbliiiing/arcface-pytorch)

