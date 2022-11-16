# CRNN


## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

```
cd PaddleOCR
pip3 install -r requirements.txt
```

## Step 2: Prepare Datasets
Download [data_lmdb_release](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here).

## Step 3: Training
Notice: modify configs/rec/rec_mv3_none_bilstm_ctc.yml file, modify the datasets path as yours.
```
cd PaddleOCR
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml Global.use_visualdl=True
```

## Reference
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
