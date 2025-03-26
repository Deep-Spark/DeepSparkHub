# Transformer

## Model Description

The Transformer model, introduced in 2017 by Vaswani et al., revolutionized natural language processing (NLP). Unlike
traditional models like RNNs or LSTMs, the Transformer relies entirely on self-attention mechanisms to process input
data in parallel, rather than sequentially. This allows it to capture long-range dependencies more effectively and scale
efficiently to large datasets. The model consists of an encoder-decoder architecture, where both components use
multi-head attention and position-wise feed-forward networks. Transformers have become the foundation for many
state-of-the-art models like BERT, GPT, and T5, driving advancements in machine translation, text generation, and other
NLP tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.3.0     |  22.12  |

## Model Preparation

### Install Dependencies

```sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/
pip3 install -r requirements.txt
```

## Model Training

The training is use AMP model.

```sh
cd PaddleNLP/examples/machine_translation/transformer
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 train.py --config ./configs/transformer.big.yaml \
--use_amp True --amp_level O1
```

## References

- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
