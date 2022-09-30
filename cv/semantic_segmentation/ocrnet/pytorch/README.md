# OCRNet

## Model description 

Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation 
It presents a simple yet effective approach, object-contextual representations, characterizing a pixel by exploiting the representation of the corresponding object class.
First, we learn object regions under the supervision of ground-truth segmentation.
Second, we compute the object region representation by aggregating the representations of the pixels lying in the object region. 
Last, the representation similarity we compute the relation between each pixel and each object region and augment the representation of each pixel with the object-contextual representation which is a weighted aggregation of all the object region representations according to their relations with the pixel. 

## Step 1: Installing
### Datasets

- download cityscape from official urls
[Cityscapes](https://www.cityscapes-dataset.com/)


- when done data folder looks like
````bash
data/
├── cityscapes
    ├── gtFine
    │   ├── test
    │   ├── train
    │   └── val
    └── leftImg8bit
    │   ├── test
    │   ├── train
    │   └── val
    ├── test.lst
    ├── trainval.lst
    └── val.lst
````

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
