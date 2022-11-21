# Wave-MLP

## Model description

In the field of computer vision, recent works show that a pure MLP architecture mainly stacked by fully-connected layers can achieve competing performance with CNN and transformer. An input image of vision MLP is usually split into multiple tokens (patches), while the existing MLP models directly aggregate them with fixed weights, neglecting the varying semantic information of tokens from different images. To dynamically aggregate tokens, we propose to represent each token as a wave function with two parts, amplitude and phase. Amplitude is the original feature and the phase term is a complex value changing according to the semantic contents of input images. Introducing the phase term can dynamically modulate the relationship between tokens and fixed weights in MLP. Based on the wave-like token representation, we establish a novel Wave-MLP architecture for vision tasks. Extensive experiments demonstrate that the proposed Wave-MLP is superior to the state-of-the-art MLP architectures on various vision tasks such as image classification, object detection and semantic segmentation. The source code is available at https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch and https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp.

## Step 1: Installing packages
```
pip install thop timm==0.4.5 torchprofile
```

## Step 2: Training

### WaveMLP_T*:

### Multiple GPUs on one machine

```
bash run.sh /home/datasets/cv/ImageNet/
```

## Results on BI-V100

### FP16

| card-batchsize-AMP opt-level | 1 card | 8 cards |
| :-----| ----: | :----: |
| BI-bs126-O1 | 114.76 | 884.27 |


### FP32

| batch_size | 1 card | 8 cards |
| :-----| ----: | :----: |
| 128 | 140.48 | 1068.15 |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 80.1                 | SDK V2.2,bs:256,8x,fp32                  | 1026        | 83.1     | 198\*8     | 0.98        | 29.4\*8                 | 1         |


## Reference
https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/wavemlp_pytorch/
