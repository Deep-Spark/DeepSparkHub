# OCRNet

## Model description

 We present a simple yet effective approach, object-contextual representations, characterizing a pixel by exploiting the representation of the corresponding object class. First, we learn object regions under the supervision of ground-truth segmentation. Second, we compute the object region representation by aggregating the representations of the pixels lying in the object region. Last, % the representation similarity we compute the relation between each pixel and each object region and augment the representation of each pixel with the object-contextual representation which is a weighted aggregation of all the object region representations according to their relations with the pixel.

## Step 1: Installation

```bash
git clone -b release/2.8 https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3
pip3 install urllib3==1.26.13
python3 setup.py install

yum install -y mesa-libGL
```

## Step 2: Preparing datasets

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure sholud look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

```bash
# Datasets preprocessing
pip3 install cityscapesscripts

python3 tools/data/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8

python3 tools/data/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","

# CityScapes PATH as follow:
tree -L 2 /path/to/cityscapes
/path/to/cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── test.txt
├── train.txt
└── val.txt
```

## Step 3: Training

```bash
# Change '/path/to/cityscapes' as your local Cityscapes dataset path
data_dir=/path/to/cityscapes
sed -i "s#: data/cityscapes#: ${data_dir}#g" configs/_base_/cityscapes.yml

# Set up environment variables
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True

# One GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml --do_eval --use_vdl


# Four GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
       --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml  \
       --do_eval \
       --use_vdl
```

## Results
| GPU         | IPS              | ACC         |
|:-----------:|:----------------:|:-----------:|
| BI-V100 x 4 | 2.12 samples/sec | mIoU=0.8120 |

## Reference
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
