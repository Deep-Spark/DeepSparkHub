# Mask2Former

## Model description

Mask2Former adopts the same meta architecture as MaskFormer, with our proposed Transformer decoder replacing the standard one. The key components of our Transformer decoder include a masked attention operator, which extracts localized features by constraining cross-attention to within the foreground region of the predicted mask for each query, instead of attending to the full feature map. To handle small objects, we propose an efficient multi-scale strategy to utilize high-resolution features. It feeds successive feature maps from the pixel decoder’s feature pyramid into successive Transformer decoder layers in a round-robin fashion. Finally, we incorporate optimization improvements that boost model performance without introducing additional computation.

## Step 1: Installation

```bash
# Install mesa-libGL
yum install mesa-libGL -y

# Install requirements
pip3 install urllib3==1.26.6
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@d779ea63faa54fe42b9b4c280365eaafccb280d6'
pip3 install cityscapesscripts

pip3 install -r requirements.txt

cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

## Step 2: Preparing datasets

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure should look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

## Step 3: Training

```bash
DETECTRON2_DATASETS=/path/to/cityscapes/ python3 train_net.py --num-gpus 8 --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml 1> train_mask2former.log 2> train_mask2former_error.log & tail -f train_mask2former.log
```

## Results

|GPUs|    fps       |       IoU Score Average   | nIoU Score Average    |
| ---           | ---                       | ---                   | ---               |
|    BI-V100×8 |  11.52                | 0.795            |  0.624       |

## Reference

- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
