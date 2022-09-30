# Faster R-CNN

## Model description

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.

## Step 1: Installing packages
```
cd <project_path>/start_scripts
bash init_torch.sh
```

## Step 2: Preparing datasets

```
$ mkdir -p <project_path>/datasets/coco
$ cd <project_path>/datasets/coco
$ wget http://images.cocodataset.org/zips/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip annotations_trainval2017.zip
$ unzip train2017.zip
$ unzip val2017.zip
```

## Step 3: Training

### On single GPU (AMP)
```
cd <project_path>/start_scripts
bash train_fasterrcnn_resnet50_amp_torch.sh
```

### Multiple GPUs on one machine
```
cd <project_path>/start_scripts
bash train_fasterrcnn_resnet50_amp_dist_torch.sh
```

## Reference
https://github.com/pytorch/vision