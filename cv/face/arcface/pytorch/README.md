# arcface
## Model description
This repo is a pytorch implement of ArcFace, which propose an Additive Angular Margin Loss to obtain highly discriminative features for face recognition. The proposed ArcFace has a clear geometric interpretation due to the exact correspondence to the geodesic distance on the hypersphere. ArcFace consistently outperforms the state-of-the-art and can be easily implemented with negligible computational overhead

## install
```bash
pip3 install requiresments.txt
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
```
## download

```bash
cd datasets
```
download dataset in this way: 
download link: https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw   password: bcrq

## preprocess

```bash
python3 txt_annotation.py
```

## train

```bash
bash run.sh $GPUS
```

## result

|   model |    FPS | LFW_Accuracy     |
|---------|--------| -----------------|
| arcface | 38.272 | 0.99000+-0.00615 |
