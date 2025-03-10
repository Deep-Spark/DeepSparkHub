# VNet

## Model Description

VNet is a fully convolutional neural network specifically designed for volumetric medical image segmentation. It extends
traditional 2D segmentation approaches to 3D, effectively processing volumetric data like CT and MRI scans. The
architecture incorporates residual connections and volumetric convolutions to capture spatial context in three
dimensions. VNet's innovative design enables precise segmentation of complex anatomical structures, making it
particularly valuable in medical imaging tasks such as organ segmentation and tumor detection in volumetric datasets.

## Model Preparation

### Prepare Resources

```bash
python3  download_dataset.py --data_dir ./data
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

```bash
# single card
python3 examples/vnet_train_and_evaluate.py --gpus 1 --batch_size 8 --base_lr 0.0001 --data_dir ./data/Task04_Hippocampus/ --model_dir ./model_train/

# 8 cards
python3 examples/vnet_train_and_evaluate.py --gpus 8 --batch_size 8 --base_lr 0.0001 --data_dir ./data/Task04_Hippocampus/ --model_dir ./model_train/
```

## Model Results

| Model | Type       | background_dice | anterior_dice | posterior_dice |
|-------|------------|-----------------|---------------|----------------|
| VNet  | multi_card | 0.9912699       | 0.83743376    | 0.81537557     |
