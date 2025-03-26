# CRNN

## Model Description

CRNN (Convolutional Recurrent Neural Network) is an end-to-end trainable model for image-based sequence recognition,
particularly effective for scene text recognition. It combines convolutional layers for feature extraction with
recurrent layers for sequence modeling, followed by a transcription layer. CRNN handles sequences of arbitrary lengths
without character segmentation or horizontal scaling, making it versatile for both lexicon-free and lexicon-based text
recognition tasks. Its compact architecture and unified framework make it practical for real-world applications like
document analysis and OCR.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.3.0     |  22.12  |

## Model Preparation

### Prepare Resources

Download [data_lmdb_release](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here).

### Install Dependencies

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git

cd PaddleOCR/
pip3 install -r requirements.txt
```

## Model Training

Notice: modify configs/rec/rec_mv3_none_bilstm_ctc.yml file, modify the datasets path as yours.

```bash
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml Global.use_visualdl=True
```

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
