# Swin Transformer
## Model description
The Swin Transformer is a type of Vision Transformer. It builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red). It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks.

## Step 1: Installing
1.Download and extract the imagenet
  - Download the datasets in the data site: [imagenet](https://image-net.org/download.php);
2.Install python requirments;
  - pip install timm==0.4.12
  - pip install yacs

## Step 2: Training
### Multiple GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128
```

## Reference
https://github.com/microsoft/Swin-Transformer
