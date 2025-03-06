# basicVSR++ (CVPR2022, Oral)

## Model description

A recurrent structure is a popular framework choice for the task of video super-resolution. The state-of-the-art method BasicVSR adopts bidirectional propagation with feature alignment to effectively exploit information from the entire input video. In this study, we redesign BasicVSR by proposing second-order grid propagation and flow-guided deformable alignment. We show that by empowering the recurrent framework with the enhanced propagation and alignment, one can exploit spatiotemporal information across misaligned video frames more effectively. The new components lead to an improved performance under a similar computational constraint. In particular, our model BasicVSR++ surpasses BasicVSR by 0.82 dB in PSNR with similar number of parameters. In addition to video super-resolution, BasicVSR++ generalizes well to other video restoration tasks such as compressed video enhancement. In NTIRE 2021, BasicVSR++ obtains three champions and one runner-up in the Video Super-Resolution and Compressed Video Enhancement Challenges. Codes and models will be released to MMEditing.

## Step 1: Installing packages

```shell
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/open-mmlab/mmagic.git -b v1.2.0 --depth=1
cd mmagic/
pip3 install -e . -v

sed -i 's/diffusers.models.unet_2d_condition/diffusers.models.unets.unet_2d_condition/g' mmagic/models/editors/vico/vico_utils.py
pip install albumentations
```

## Step 2: Preparing datasets

Download REDS dataset from [homepage](https://seungjunnah.github.io/Datasets/reds.html)
```shell
mkdir -p data/
ln -s ${REDS_DATASET_PATH} data/REDS
```

## Step 3: Training

### One single GPU
```shell
python3 tools/train.py configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py
```

### Mutiple GPUs on one machine
```shell
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py 8
```

## Reference
[mmagic](https://github.com/open-mmlab/mmagic)
