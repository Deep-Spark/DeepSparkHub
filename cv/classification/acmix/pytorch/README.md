# ACmix

## Model description

Convolution and self-attention are two powerful techniques for representation learning, and they are usually considered as two peer approaches that are distinct from each other. In this paper, we show that there exists a strong underlying relation between them, in the sense that the bulk of computations of these two paradigms are in fact done with the same operation. Specifically, we first show that a traditional convolution with kernel size k x k can be decomposed into k^2 individual 1x1 convolutions, followed by shift and summation operations. Then, we interpret the projections of queries, keys, and values in self-attention module as multiple 1x1 convolutions, followed by the computation of attention weights and aggregation of the values. Therefore, the first stage of both two modules comprises the similar operation. More importantly, the first stage contributes a dominant computation complexity (square of the channel size) comparing to the second stage. This observation naturally leads to an elegant integration of these two seemingly distinct paradigms, i.e., a mixed model that enjoys the benefit of both self-Attention and Convolution (ACmix), while having minimum computational overhead compared to the pure convolution or self-attention counterpart. Extensive experiments show that our model achieves consistently improved results over competitive baselines on image recognition and downstream tasks. Code and pre-trained models will be released at https://github.com/LeapLabTHU/ACmix and https://gitee.com/mindspore/models.

## Step 1: Installing packages
```
pip install termcolor==1.1.0 yacs==0.1.8 timm==0.4.5
```


## Step 2: Training

### Swin-S + ACmix on ImageNet using 8 cards:
```
bash run.sh 8 acmix_swin_small_patch4_window7_224.yaml <DATA_DIR>
```

## Results on BI-V100

| card | batch_size | Single Card | 8 Cards |
|:-----|------------|------------:|:-------:|
| BI   |     128    |       63.59 | 502.22  |


## Reference
https://github.com/leaplabthu/acmix