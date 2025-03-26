# MODEL_NAME (Related Toolbox for LLM)

## Model Description

A brief introduction about this model.
A brief introduction about this model.
A brief introduction about this model.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.2.0     |  23.03  |

## Model Preparation

### Prepare Resources

```bash
python3 dataset/coco/download_coco.py
```

Go to huggingface.

### Install Dependencies

```bash
pip install -r requirements.txt
python3 setup.py install
```

## Model Training

```bash
# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval

# 1 GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval
```

## Model Results

| Model      | Iluvatar GPU | FPS | ACC  |
|------------|--------------|-----|------|
| MODEL_NAME | BI-V100 x1   | 32  | 71.2 |
| MODEL_NAME | BI-V100 x8   | 251 | 71.2 |

## References

- [Reference_link](https://github.com/reference_repo/reference_repo)
- [Paper](Paper_link)
