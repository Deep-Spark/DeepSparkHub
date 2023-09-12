# HRNet-W32

## Model description
HRNet (High-Resolution Net) is proposed for the task of 2D human pose estimation (Human Pose Estimation or Keypoint Detection), and the network is mainly aimed at the pose assessment of a single individual. Most existing human pose estimation methods recover high-resolution representations from low-resolution representations produced by a high-to-low resolution network. Instead, HRNet maintains high-resolution representations through the whole process. HRNet starts from a high-resolution subnetwork as the first stage, gradually add high-to-low resolution subnetworks one by one to form more stages, and connect the mutli-resolution subnetworks in parallel. Then, HRNet  conducts repeated multi-scale fusions such that each of the high-to-low resolution representations receives information from other parallel representations over and over, leading to rich high-resolution representations. As a result, the predicted keypoint heatmap is potentially more accurate and spatially more precise. 


## Step 1: Installation
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

```
cd PaddleDetection
pip3 install numba==0.56.4
pip3 install -r requirements.txt
python3 setup.py install
```

## Step 2: Preparing Datasets
```
python3 dataset/coco/download_coco.py
```
The coco2017 dataset path structure should look like:
```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── train2017.txt
├── val2017.txt
└── ...
```
## Step 3: Training

single GPU
```
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
python3 tools/train.py -c configs/keypoint/hrnet/hrnet_w32_384x288.yml  --eval -o use_gpu=true
```

8 GPU(Distributed Training)
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
python3 -m paddle.distributed.launch --log_dir=./log_hrnet_w32_384x288/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/keypoint/hrnet/hrnet_w32_384x288.yml --eval
```
## Results on BI-V100
<div align="center">

| GPU         | FP32                                               |
| ----------- | ----------------------------------------------     |
| 8 cards     | BatchSize=64,AP(coco val)=78.4, single GPU FPS= 45 |

</div>

## Reference
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
