# OCRNet

## Model description 

Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation 
It presents a simple yet effective approach, object-contextual representations, characterizing a pixel by exploiting the representation of the corresponding object class.
First, we learn object regions under the supervision of ground-truth segmentation.
Second, we compute the object region representation by aggregating the representations of the pixels lying in the object region. 
Last, the representation similarity we compute the relation between each pixel and each object region and augment the representation of each pixel with the object-contextual representation which is a weighted aggregation of all the object region representations according to their relations with the pixel. 

## Step 1: Installing
### Datasets

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure sholud look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

```bash
mkdir data/
ln -s /path/to/cityscapes data/cityscapes
```

### Environment
```bash
bash init.sh
```

## Step 2: Training
### Train using 1x GPU card
```bash
bash train.sh
```

### Train using Nx GPU cards
when using 4 x GPU  
```bash
bash train_distx4.sh
```

when using 8 x GPU  
```bash
bash train_distx8.sh
```

## Reference

Ref: https://github.com/HRNet/HRNet-Semantic-Segmentation
