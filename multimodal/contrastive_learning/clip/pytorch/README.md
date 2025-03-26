# CLIP

## Model Description

Contrastive Language-Image Pre-training (CLIP), consisting of a simplified version of ConVIRT trained from scratch, is
an efficient method of image representation learning from natural language supervision. , CLIP jointly trains an image
encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time
the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target
dataset’s classes.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Download CIFAR100.

### Install Dependencies

```sh
cd multimodal/Language-Image_Pre-Training/clip/pytorch
pip3 install ftfy regex tqdm
```

## Model Training

#### Zero-shot task on single GPU

```sh
# top5描述： CLIP执行zero-shot预测任务,从数据集CIFAR100测试集中获取图像，
# 并计算测试集中图像与文本能够匹配的TOP5的标签的accuracy
python3 clip/zero_shot_prediction_top5.py

# top1描述：CLIP执行zero-shot预测任务,从数据集CIFAR100测试集中获取图像，
# 并计算测试集中图像与文本能够匹配的TOP1的标签的accuracy
python3 clip/zero_shot_prediction_top1.py
```

#### Linear-probe evaluation task by using scikit-learn on single GPU

```sh
# 使用scikit-learn对图像特征进行逻辑回归
python3 clip/Linear_probe_evaluation.py
```

## Model Results

| Model | GPUs    | Type                      | accuracy(%) |
|-------|---------|---------------------------|-------------|
| CLIP  | BI-V100 | zero-shot-prediction-top5 | 86.74       |
| CLIP  | BI-V100 | zero-shot-prediction-top1 | 61.71       |
| CLIP  | BI-V100 | linear-probe-evaluation   | 80.01       |

## References

- [CLIP](https://github.com/openai/CLIP)
