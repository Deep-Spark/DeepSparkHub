# BlazeFace

## Model Description

BlazeFace is Google Research published face detection model. It's lightweight but good performance, and tailored for
mobile GPU inference. It runs at a speed of 200-1000+ FPS on flagship devices.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

We use WIDER-FACE dataset for training and model tests, the official web site provides detailed data is introduced.

- WIDER-Face data source:

- Load a dataset of type wider_face using the following directory structure:

```bash
dataset/wider_face/
├── wider_face_split
│   ├── wider_face_train_bbx_gt.txt
│   ├── wider_face_val_bbx_gt.txt
├── WIDER_train
│   ├── images
│   │   ├── 0--Parade
│   │   │   ├── 0_Parade_marchingband_1_100.jpg
│   │   │   ├── 0_Parade_marchingband_1_381.jpg
│   │   │   │   ...
│   │   ├── 10--People_Marching
│   │   │   ...
├── WIDER_val
│   ├── images
│   │   ├── 0--Parade
│   │   │   ├── 0_Parade_marchingband_1_1004.jpg
│   │   │   ├── 0_Parade_marchingband_1_1045.jpg
│   │   │   │   ...
│   │   ├── 10--People_Marching
│   │   │   ...
```

- Manually download the dataset: To download the WIDER-FACE dataset, run the following command:

```bash
cd dataset/wider_face && ./download_wider_face.sh
```

### Install Dependencies

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

```bash
cd PaddleDetection
yum install mesa-libGL -y

pip3 install -r requirements.txt
pip3 install protobuf==3.20.1
pip3 install urllib3==1.26.6
pip3 install IPython
pip3 install install numba==0.56.4
```

## Model Training

```bash
cd PaddleDetection

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/face_detection/blazeface_fpn_ssh_1000e.yml

# Evaluation
python3 -u tools/eval.py -c configs/face_detection/blazeface_fpn_ssh_1000e.yml \
       -o weights=output/blazeface_fpn_ssh_1000e/model_final.pdopt \
       multi_scale=True

wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip
unzip eval_tools.zip && rm -f eval_tools.zip

git clone https://github.com/wondervictor/WiderFace-Evaluation.git
cd WiderFace-Evaluation
python3 setup.py build_ext --inplace
python3 evaluation.py -p ../output/pred/ -g ../eval_tools/ground_truth
```

## Model Results

 | Model             | GPUs       | Easy/Medium/Hard Set | Ips    |
 |-------------------|------------|----------------------|--------|
 | BlazeFace-FPN-SSH | BI-V100 x8 | 0.886/0.860/0.753    | 8.6813 |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
