## Facenet
This is a facenet-pytorch library that can be used to train your own face recognition model.
## install
```bash
pip3 install requiresments.txt
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
pip3 install matplotlib
pip3 install scikit-learn
```
## download

download dataset in this way: 
download link: https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw   password: bcrq
```bash
ln -s ${your_path_to_face} datasets/
ln -s ${your_path_to_face} lfw/
ln -s ${your_path_to_face} lfw_pair.txt
```

## preprocess

```bash
python3 txt_annotation.py
```

## train

```bash
python3 train.py
```

## result

|   model |    FPS | LFW_Accuracy     |
|---------|--------| -----------------|
| facenet | 1256.96| 0.97933+-0.00624 |
