# HRNet-W32

## Model Description

HRNet, or High-Resolution Net, is a general purpose convolutional neural network for tasks like semantic segmentation,
object detection and image classification. It is able to maintain high resolution representations through the whole
process. We start from a high-resolution convolution stream, gradually add high-to-low resolution convolution streams
one by one, and connect the multi-resolution streams in parallel. The resulting network consists of several stages and
the nth stage contains n streams corresponding to n resolutions. The authors conduct repeated multi-resolution fusions
by exchanging the information across the parallel streams over and over.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

```bash
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

### Install Dependencies

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
pip3 install numba==0.56.4
pip3 install -r requirements.txt
python3 setup.py install
```

## Model Training

```bash
# single GPU
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
python3 tools/train.py -c configs/keypoint/hrnet/hrnet_w32_384x288.yml  --eval -o use_gpu=true

# 8 GPU(Distributed Training)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
python3 -m paddle.distributed.launch --log_dir=./log_hrnet_w32_384x288/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/keypoint/hrnet/hrnet_w32_384x288.yml --eval
```

## Model Results

| GPU        | FP32                                               |
|------------|----------------------------------------------------|
| BI-V100 x8 | BatchSize=64,AP(coco val)=78.4, single GPU FPS= 45 |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
