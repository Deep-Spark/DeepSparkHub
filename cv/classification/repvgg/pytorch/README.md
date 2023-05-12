# RepVGG
## Model description
 A simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. 

## Step 1: Installing

```bash
pip3 install timm yacs
```

## Step 2: Download data

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

The ImageNet dataset path structure should look like:

```bash
imagenet
├── train
│   └── n01440764
│       ├── n01440764_10026.JPEG
│       └── ...
├── train_list.txt
├── val
│   └── n01440764
│       ├── ILSVRC2012_val_00000293.JPEG
│       └── ...
└── val_list.txt
```

## Step 3: Run RepVGG

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main.py --arch [model name] --data-path [/path/to/imagenet] --batch-size 32 --tag train_from_scratch --output ./ --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 1e-4 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET weak AUG.MIXUP 0.0 DATA.DATASET imagenet DATA.IMG_SIZE 224
```

The original RepVGG models were trained in 120 epochs with cosine learning rate decay from 0.1 to 0. We used 8 GPUs, global batch size of 256, weight decay of 1e-4 (no weight decay on fc.bias, bn.bias, rbr_dense.bn.weight and rbr_1x1.bn.weight) (weight decay on rbr_identity.weight makes little difference, and it is better to use it in most of the cases), and the same simple data preprocssing as the PyTorch official example:

```py
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```

The valid model names include (--arch [model name])

```
RepVGGplus-L2pse, RepVGG-A0, RepVGG-A1, RepVGG-A2, RepVGG-B0, RepVGG-B1, RepVGG-B1g2, RepVGG-B1g4, RepVGG-B2, RepVGG-B2g2, RepVGG-B2g4, RepVGG-B3, RepVGG-B3g2, RepVGG-B3g4
```

|   model  |     GPU     | FP32                                 | 
|----------| ----------- | ------------------------------------ |
| RepVGG-A0| 8 cards     | Acc@1=0.7241                         |







