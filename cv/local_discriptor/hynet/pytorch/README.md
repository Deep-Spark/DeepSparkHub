# HyNet

## Model description

A new local descriptor that leads to stateof-the-art results in matching. HyNet introduces a hybrid similarity measure for triplet margin loss, a regularisation term constraining the descriptor norm, and a new network architecture that performs L2 normalisation of all intermediate feature maps and the output descriptors. HyNet surpasses previous methods by a significant margin on standard benchmarks that include patch matching, verification, and retrieval, as well as outperforming full end-to-end methods on 3D reconstruction tasks.


## Step 1: Installing
### Install packages
```
pip3 install opencv-python PIL tqdm
```

### Prepare datasets
```
mkdir -p data/
cd data

wget http://matthewalunbrown.com/patchdata/liberty.zip
wget http://matthewalunbrown.com/patchdata/notredame.zip
wget http://matthewalunbrown.com/patchdata/yosemite.zip

unzip liberty.zip -d liberty
unzip notredame.zip -d notredame
unzip yosetime.zip -d yosetime
```

## Step 2: Training

### single GPU

#### extracting HyNet descriptor without SOS regularization
```
python3 train.py --lr 0.01 --train_set liberty --desc_name HyNet --is_sosr False
```

#### extracting HyNet descriptor with SOS regularization
```
python3 train.py --lr 0.01 --train_set liberty --desc_name HyNet --is_sosr True
```

## Results on BI-V100

## Reference

Ref: https://github.com/yuruntian/HyNet

