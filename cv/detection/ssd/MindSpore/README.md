# SSD
## Model description
SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.

[Paper](https://arxiv.org/abs/1512.02325):   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).
## Step 1: Installing
```
pip3 install -r requirements.txt
pip3 install easydict
```
## Step 2: Prepare Datasets
download dataset in /home/datasets/cv/coco2017

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - Note：Data will be processed in dataset.py

  Change the `coco_root` and other settings you need in `src/config.py`. The directory structure is as follows:

  ```shell
  .
  └─coco_dataset
    ├─annotations
      ├─instance_train2017.json
      └─instance_val2017.json
    ├─val2017
    └─train2017
  ```
  If your own dataset is used. **Select dataset to other when run script.**
      Organize the dataset information into a TXT file, each row in the file is as follows:

      ```shell
      train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
      ```

      Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `src/config.py`.
# [Pretrained models](#contents)
Please [resnet50](https://pan.baidu.com/s/1rrhsZqDVmNxR-bCnMPvFIw?pwd=8766) download resnet50.ckpt here
```
mv resnet50.ckpt ./ckpt
```

## Step 3: Training
```
mpirun -allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
python3 train.py \
--run_distribute=True \
--lr=0.05 \
--dataset=coco \
--device_num=8 \
--loss_scale=1 \
--device_target="GPU" \
--epoch_size=60 \
--config_path=./config/ssd_resnet50_fpn_config_gpu.yaml \
--output_path './output' > log.txt 2>&1 &
```
### [Evaluation result]
## Results on BI-V100

| GPUs | per step time  |  MAP  |
|------|--------------  |-------|
|  1*8 |   0.814s       | 0.374 |

## Results on NV-V100s

| GPUs | per step time  |  MAP  |
|------|--------------  |-------|
|  1*8 |   0.797s       | 0.369 |