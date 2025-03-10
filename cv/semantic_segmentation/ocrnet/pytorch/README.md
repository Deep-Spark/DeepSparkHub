# OCRNet

## Model Description

OCRNet (Object Contextual Representation Network) is a deep learning model for semantic segmentation that enhances
pixel-level understanding by incorporating object context information. It learns object regions from ground-truth
segmentation and aggregates pixel representations within these regions. By computing relationships between pixels and
object regions, OCRNet augments each pixel's representation with contextual information from relevant objects. This
approach improves segmentation accuracy, particularly in complex scenes, by better capturing object boundaries and
contextual relationships between different image elements.

## Model Preparation

### Prepare Resources

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the
Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure
sholud look like:

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

### Install Dependencies

```bash
bash init.sh
```

## Model Training

```bash
# Train using 1x GPU card
bash train.sh

# Train using 4x GPU cards
bash train_distx4.sh

# Training using 8 x GPU  
bash train_distx8.sh
```

## References

- [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
