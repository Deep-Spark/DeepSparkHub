# Swin Transformer

## Setup 
1.Download and extract the imagenet
  - Download the datasets in the data site: [imagenet](https://image-net.org/download.php);
2.Install python requirments;
  - pip install timm==0.4.12
  - pip install yacs

## Training 
`python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 `

## Performace
|     card type    | FPS(1卡)   | FPS(8卡)  |
| --------   | -----:  | :----:  |
| V100S     | 310.802   |   1992.264     |
| BI      |   145.503   |   1073.904  |
