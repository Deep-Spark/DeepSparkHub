# BLIP

## Model Description

BLIP (Bootstrapped Language-Image Pretraining) is an innovative vision-language model that excels in both understanding
and generation tasks. Unlike traditional models that specialize in one area, BLIP effectively bridges the gap between
visual comprehension and text generation. It employs a unique bootstrapping mechanism to filter and enhance noisy
web-sourced image-text pairs, improving the quality of training data. This approach enables BLIP to achieve superior
performance in tasks like image captioning, visual question answering, and multimodal understanding.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO2014 dataset.

The COCO2014 dataset path structure should look like:

```sh
coco2014
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   └── ...
├── train2014
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
│   └── ...
├── val2014
│   ├── COCO_val2014_000000000042.jpg
│   ├── COCO_val2014_000000000073.jpg
│   └── ...
├── labels
│   ├── train2014
│       ├── COCO_train2014_000000000009.txt
│       ├── COCO_train2014_000000000025.txt
│       └── ... 
│   └── val2014
|       ├── COCO_val2014_000000000042.txt
│       ├── COCO_val2014_000000000073.txt
│       └── ... 
```

### Install Dependencies

```sh
yum install mesa-libGL
yum install -y java-1.8.0-openjdk
git clone https://github.com/salesforce/BLIP.git
cd BLIP
pip3 install -r requirements.txt
pip3 install ruamel_yaml
pip3 install urllib3==1.26.6
```

## Model Training

### 8 GPUs on one machine

Set 'image_root' in configs/caption_coco.yaml to '/path/to/coco2014/'

```sh
rm -rf train_caption.py
mv ../train_caption.py .
python3 -m torch.distributed.run --nproc_per_node=8 train_caption.py 
```

### Evaluation

Set 'pretrained' in configs/caption_coco.yaml to 'output/Caption_coco/checkpoint_best.pth'

```sh
python3 -m torch.distributed.run --nproc_per_node=8 train_caption.py --evaluate
```

## Model Results

| Model | GPUs      | Bleu score                                                 | Training performance |
|-------|-----------|------------------------------------------------------------|----------------------|
| BLIP  | BI V100×8 | Bleu_1: 0.797, Bleu_2: 0.644,Bleu_3: 0.503, ,Bleu_4: 0.388 | 1.9790 s / it        |

## References

- [BLIP](https://github.com/salesforce/BLIP)
- [Paper](https://proceedings.mlr.press/v162/li22n/li22n.pdf)
