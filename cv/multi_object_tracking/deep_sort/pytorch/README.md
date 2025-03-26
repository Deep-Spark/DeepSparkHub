# DeepSORT

## Model Description

DeepSORT is an advanced multi-object tracking algorithm that extends SORT by incorporating deep learning-based
appearance features. It combines motion information with a CNN-based RE-ID model to track objects more accurately,
especially in complex scenarios with occlusions. DeepSORT uses a Kalman filter for motion prediction and associates
detections using both motion and appearance cues. This approach improves tracking consistency and reduces identity
switches, making it particularly effective for person tracking in crowded scenes and video surveillance applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Download the [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

The original data set structure is as follows:

```sh
Market-1501-v15.09.15
├── bounding_box_test
├── bounding_box_train
├── gt_bbox
├── gt_query
├── query
└── readme.txt
```

We need to generate train and test datasets.

```sh
python3 create_train_test_datasets.py --origin_datasets_path origin_datasets_path --datasets_path process_datasets_path
```

We need to generate query and gallery datasets for evaluate.

```sh
python3 create_query_gallery_datasets.py --origin_datasets_path origin_datasets_path --datasets_path process_datasets_path
```

After the datasets is processed, the datasets structure is as follows:

```sh
data
├── train
├── test
├── query
├── gallery
```

## Model Training

The original model used in paper is in original_model.py, and its parameter here
[original_ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).  

```sh
# Train
python3 train.py --data-dir your path

# Evaluate your model.
python3 test.py --data-dir your path
python3 evaluate.py
```

## Model Results

| Model    | GPU        | Top1 ACC |
|----------|------------|----------|
| DeepSORT | BI-V100 x8 | 0.980    |

## References

- [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
