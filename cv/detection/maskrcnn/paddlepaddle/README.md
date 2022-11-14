# Mask R-CNN

## Model description

Nuclei segmentation is both an important and in some ways ideal task for modern computer vision methods, e.g. convolutional neural networks. While recent developments in theory and open-source software have made these tools easier to implement, expert knowledge is still required to choose the right model architecture and training setup. We compare two popular segmentation frameworks, U-Net and Mask-RCNN in the nuclei segmentation task and find that they have different strengths and failures. To get the best of both worlds, we develop an ensemble model to combine their predictions that can outperform both models by a significant margin and should be considered when aiming for best nuclei segmentation performance.

## Step 1: Installing

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip3 install -r requirements.txt
python3 setup.py install --user
```

## Step 2: Download data

Download the [COCO Dataset](https://cocodataset.org/#home) 

```bash

#COCO PATH as follow:
#ls -al /home/datasets/coco
#total 5208
#drwxrwxr-x 6 1003 1003      93 Dec 29  2021 .
#drwxr-xr-x 6 root root     179 Jul 18 06:48 ..
#drwxrwxr-x 2 1003 1003     322 Sep 24  2021 annotations
#drwxrwxr-x 2 1003 1003      54 Dec 29  2021 pkl_coco
#drwxrwxr-x 2 1003 1003 3846144 Sep 24  2021 train2017
#drwxrwxr-x 2 1003 1003  163840 Sep 24  2021 val2017

```

## Step 3: Run Mask R-CNN

```bash
# Make sure your dataset path is the same as above
#coco_dir=${coco_dir:-/home/datasets/coco/}
sed -i "s#dataset/coco#${coco_dir}#g" configs/datasets/coco_instance.yml
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml --use_vdl=true --eval --amp
```