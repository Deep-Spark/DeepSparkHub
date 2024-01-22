# BLIP
> [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n/li22n.pdf)

## Abstract

Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. 

## Step 1: Installation

```bash
yum install mesa-libGL
yum install -y java-1.8.0-openjdk
git clone https://github.com/salesforce/BLIP.git
cd BLIP
pip3 install -r requirements.txt
pip3 install ruamel_yaml
pip3 install urllib3==1.26.6
```
## Step 2: Preparing datasets
Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO2014 dataset.

The COCO2014 dataset path structure should look like:
```bash
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

## Step 3: Training

### 8 GPUs on one machine
Set 'image_root' in configs/caption_coco.yaml to '/path/to/coco2014/'
```bash
rm -rf train_caption.py
mv ../train_caption.py .
python3 -m torch.distributed.run --nproc_per_node=8 train_caption.py 
```
## Step 4: Evaluting

Set 'pretrained' in configs/caption_coco.yaml to 'output/Caption_coco/checkpoint_best.pth'
```bash
python3 -m torch.distributed.run --nproc_per_node=8 train_caption.py --evaluate
```

## Results

| GPUS      |    Bleu score                                              | train performance|
| ----------| -----------------------------------------------------------|------------------|
| BI V100×8 |  Bleu_1: 0.797, Bleu_2: 0.644,Bleu_3: 0.503, ,Bleu_4: 0.388|   1.9790 s / it  |

## Reference
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://github.com/salesforce/BLIP)