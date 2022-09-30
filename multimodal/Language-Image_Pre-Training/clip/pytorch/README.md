# CLIP

## Model description

Contrastive Language-Image Pre-training (CLIP), consisting of a simplified version of ConVIRT trained from scratch, is an efficient method of image representation learning from natural language supervision. , CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes.


## Step 1: Installing packages

```shell
cd multimodal/Language-Image_Pre-Training/clip/pytorch
pip3 install ftfy regex tqdm pytorch torchvision
```

## Step 2: Preparing datasets

Download CIFAR100

## Step 3: Training

### Zero-shot task

<!-- top5描述： CLIP执行zero-shot预测任务,从数据集CIFAR100测试集中获取图像，并计算测试集中图像与文本能够匹配的TOP5的标签的accuracy -->

#### On single GPU

```
python3 clip/zero_shot_prediction_top5.py
```

<!--top1描述：CLIP执行zero-shot预测任务,从数据集CIFAR100测试集中获取图像，并计算测试集中图像与文本能够匹配的TOP1的标签的accuracy -->

```
python3 clip/zero_shot_prediction_top1.py
```

### Linear-probe evaluation task by using scikit-learn

<!-- 使用scikit-learn对图像特征进行逻辑回归 -->

#### On single GPU

```
python3 clip/Linear_probe_evaluation.py
```

## Results on BI-V100

* zero-shot-prediction-top5

|   metric    | BI-V100 |
| ----------- | ------- |
| accuracy(%) | 86.74   |


* zero-shot-prediction-top1


|   metric    | BI-V100 |
| ----------- | ------- |
| accuracy(%) | 61.71   |



* linear-probe-evaluation

|   metric    | BI-V100 |
| ----------- | ------- |
| accuracy(%) | 80.01   |


## Reference
https://github.com/openai/CLIP