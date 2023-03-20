## Introduction
This is an implement of MOT tracking algorithm deep sort. Deep sort is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. This CNN model is indeed a RE-ID model and the detector used in [PAPER](https://arxiv.org/abs/1703.07402) is FasterRCNN , and the original source code is [HERE](https://github.com/nwojke/deep_sort).  
However in original code, the CNN model is implemented with tensorflow, which I'm not familier with. SO I re-implemented the CNN feature extraction model with PyTorch, and changed the CNN model a little bit. Also, I use **YOLOv3** to generate bboxes instead of FasterRCNN.

We just need to train the RE-ID model!

## Preparing datasets
Download the [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) 

The original data set structure is as follows:
```
Market-1501-v15.09.15
├── bounding_box_test
├── bounding_box_train
├── gt_bbox
├── gt_query
├── query
└── readme.txt
```

We need to generate train and test datasets.

```
python3 create_train_test_datasets.py --origin_datasets_path origin_datasets_path --datasets_path process_datasets_path
```
We need to generate query and gallery datasets for evaluate.
```
python3 create_query_gallery_datasets.py --origin_datasets_path origin_datasets_path --datasets_path process_datasets_path
```

After the datasets is processed, the datasets structure is as follows:
```
data
├── train
├── test
├── query
├── gallery
```

## Training
The original model used in paper is in original_model.py, and its parameter here [original_ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).  

```
python3 train.py --data-dir your path
```

## Evaluate your model
```
python3 test.py --data-dir your path
python3 evaluate.py
```
## Results
Acc top1:0.980

## Quick Start All Processes
Please refer to https://github.com/ZQPei/deep_sort_pytorch
