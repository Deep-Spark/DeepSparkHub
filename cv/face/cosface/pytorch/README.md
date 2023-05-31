# Cosface
## Model description
This project is aimmed at implementing the CosFace described by the paper CosFace: Large Margin Cosine Loss for Deep Face Recognition. 

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
download dataset in this way: download link: https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw password: bcrq

## preprocess

```bash
python3 txt_annotation.py
```

## train
### single gpu
```bash
bash train.sh 0
```
### multi gpus
```bash
bash train.sh 0,1,2,3,4,5,6,7
```

## result

|   model |    FPS | LFW_Accuracy     |
|---------|--------| -----------------|
| Cosface | 5492.08 | 0.9865          |


